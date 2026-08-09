//! Chat Completions passthrough with per-session prefix substitution.
//!
//! Chat Completions is stateless: the client resends the whole conversation on
//! every call. When a request carries a session header, the leading messages
//! this gateway has already folded are replaced with a stored stand-in, so the
//! prompt sent upstream stops growing with every turn.
//!
//! The client is unaffected — it gets an ordinary Chat Completions response and
//! keeps its own untouched history.

use axum::extract::{Request, State};
use axum::response::Response;
use bytes::Bytes;
use http::HeaderMap;
use serde_json::Value;
use tracing::{debug, info, warn};

use agentic_core::proxy::{ProxyAuth, ProxyBody, ProxyRequest, proxy_request_with_path};

use super::super::common::{convert_response, read_bytes};
use crate::app::AppState;
use crate::compaction;

/// Matches the default key of llm-d's session-affinity scorer, so one value
/// serves both this prefix store and upstream endpoint stickiness.
const SESSION_HEADER: &str = "x-session-id";

const UPSTREAM_PATH: &str = "/v1/chat/completions";

pub async fn chat_completions(State(state): State<AppState>, req: Request) -> Response {
    let (parts, body) = req.into_parts();
    let bytes = match read_bytes(body).await {
        Ok(bytes) => bytes,
        Err(response) => return response,
    };

    // Parsed once; everything below reads from this.
    let request: Option<Value> = serde_json::from_slice(&bytes).ok();
    let messages = request.as_ref().and_then(|body| body["messages"].as_array()).cloned();
    let session = session_id(&parts.headers);

    let mut upstream = bytes.clone();
    if let (Some(session), Some(request), Some(messages)) = (&session, &request, &messages) {
        if let Some(rewritten) = substitute(&state, session, request, messages).await {
            upstream = rewritten;
        }
    }

    // `reply_body` is None for a streamed response, so streaming requests
    // substitute but never fold — the reply only exists as SSE chunks.
    let (response, reply_body) = forward(&state, parts.headers, parts.uri.query(), upstream).await;

    let (Some(session), Some(request), Some(messages), Some(reply_body)) = (session, request, messages, reply_body)
    else {
        return response;
    };
    let Some(reply) = assistant_message(&reply_body) else {
        debug!("no assistant message in reply for {session}: skipping fold");
        return response;
    };

    spawn_fold(&state, session, request["model"].as_str().unwrap_or_default().to_owned(), messages, reply);
    response
}

fn session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get(SESSION_HEADER)
        .and_then(|value| value.to_str().ok())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

/// Rewrite the request with this session's stored prefix.
///
/// `None` means "send the original bytes" — no stored prefix, a lookup failure,
/// or a history too short to substitute into.
async fn substitute(state: &AppState, session: &str, request: &Value, messages: &[Value]) -> Option<Bytes> {
    let prefix = match state.exec_ctx.session_prefix_store().get(session).await {
        Ok(prefix) => prefix?,
        Err(error) => {
            // A lookup failure must not fail the request.
            warn!("prefix lookup failed for {session}: {error}");
            return None;
        }
    };

    // A client that branched or regenerated can send fewer messages than the
    // prefix accounts for. Substituting would drop the whole array, including
    // the new user message.
    if prefix.replaced_count >= messages.len() {
        debug!(
            "stored prefix for {session} covers {} of {} messages: forwarding unchanged",
            prefix.replaced_count,
            messages.len()
        );
        return None;
    }

    let mut rewritten = prefix.replacement;
    rewritten.extend_from_slice(&messages[prefix.replaced_count..]);
    info!(
        session = %session,
        replaced = prefix.replaced_count,
        upstream_messages = rewritten.len(),
        client_messages = messages.len(),
        "substituted stored prefix"
    );

    let mut body = request.clone();
    *body.get_mut("messages")? = Value::Array(rewritten);
    serde_json::to_vec(&body).ok().map(Bytes::from)
}

/// Forward upstream, also returning the reply body when it was read in full.
async fn forward(
    state: &AppState,
    headers: HeaderMap,
    query: Option<&str>,
    body: Bytes,
) -> (Response, Option<Bytes>) {
    let request = ProxyRequest {
        headers,
        body,
        query: query.map(str::to_owned),
    };
    let proxied = proxy_request_with_path(request, UPSTREAM_PATH, ProxyAuth::OpenAiBearer, &state.proxy_state).await;

    let reply = match &proxied.body {
        ProxyBody::Full(bytes) if proxied.status.is_success() => Some(bytes.clone()),
        _ => None,
    };
    (convert_response(proxied), reply)
}

fn assistant_message(reply: &Bytes) -> Option<Value> {
    let body: Value = serde_json::from_slice(reply).ok()?;
    body["choices"].as_array()?.first()?.get("message").cloned()
}

/// Fold this turn into the session's prefix, off the request path.
///
/// With no compaction service configured — or when it fails — the history is
/// stored unchanged, which costs compression but never correctness.
fn spawn_fold(state: &AppState, session: String, model: String, mut messages: Vec<Value>, reply: Value) {
    messages.push(reply);
    let store = state.exec_ctx.session_prefix_store();
    let client = std::sync::Arc::clone(&state.exec_ctx.client);
    let address = state.compaction_address.clone();

    tokio::spawn(async move {
        let replaced = messages.len();
        let replacement = match address.as_deref() {
            Some(address) => compaction::compact(address, &client, &model, &messages)
                .await
                .unwrap_or(messages),
            None => messages,
        };
        info!(session = %session, replaced, stored = replacement.len(), "folded session prefix");

        if let Err(error) = store.upsert(&session, replaced, &replacement).await {
            warn!("failed to store prefix for {session}: {error}");
        }
    });
}

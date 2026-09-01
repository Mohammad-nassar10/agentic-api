//! The endpoints, and the axum glue they need. Both read and write conversation
//! history, so `/internal` requires a shared bearer token; the probes do not.
//! Keep the listener cluster-internal regardless - the token identifies nobody,
//! so it authenticates the caller without authorizing it per tenant.

use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::value::RawValue;
use tracing::warn;

use agentic_core::executor::request::RequestContext;

use agentic_core::executor::{
    ExecutorError, UpstreamBody, commit, decode_upstream, rehydrate_conversation, upstream_request,
};
use agentic_core::types::request_response::RequestPayload;

use crate::InternalState;
use crate::context::{Hydration, ensure_splittable, seal, unseal};

const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;
/// Readiness means storage answers - llm-d owns the model fleet.
const STORAGE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Body of `POST /internal/persist`: the context, plus exactly one response form.
#[derive(Debug, Deserialize)]
pub struct PersistRequest {
    context: String,
    response: Option<Box<RawValue>>,
    sse: Option<String>,
}

/// Rejects any `/internal` call that does not present the shared secret. These
/// endpoints read and write conversation history, so an unauthenticated caller
/// must not reach them; the probes are layered separately and stay open.
pub async fn require_token(State(state): State<InternalState>, request: Request, next: Next) -> Response {
    let presented = request
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "));
    match presented {
        Some(token) if token_matches(token, &state.api_token) => next.run(request).await,
        _ => json(
            StatusCode::UNAUTHORIZED,
            br#"{"error":{"type":"invalid_request_error","message":"missing or invalid bearer token"}}"#.to_vec(),
        ),
    }
}

/// Compared without an early return, so a wrong token takes the same time
/// whatever its first differing byte.
fn token_matches(presented: &str, expected: &str) -> bool {
    presented.len() == expected.len()
        && presented
            .bytes()
            .zip(expected.bytes())
            .fold(0_u8, |differences, (a, b)| differences | (a ^ b))
            == 0
}

pub async fn health() -> StatusCode {
    StatusCode::OK
}

pub async fn ready(State(state): State<InternalState>) -> StatusCode {
    if state.exec_ctx.storage_ready(STORAGE_PROBE_TIMEOUT).await {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

pub async fn internal_hydrate(State(state): State<InternalState>, req: Request) -> Response {
    let payload: RequestPayload = match read_json(req.into_body()).await {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    match hydrate(payload, &state).await {
        Ok(hydration) => axum::Json(hydration).into_response(),
        Err(error) => error_response(error),
    }
}

/// Rehydrates the turn and builds the request the caller forwards to a model.
#[allow(clippy::result_large_err)] // `ExecutorError` is core's; boxing it is not ours to decide
async fn hydrate(request: RequestPayload, state: &InternalState) -> agentic_core::executor::ExecutorResult<Hydration> {
    ensure_splittable(&request)?;
    let ctx = rehydrate_conversation(request, state.exec_ctx.as_ref()).await?;
    let stream = ctx.original_request.stream;
    let request = RawValue::from_string(upstream_request(&ctx, stream)?).map_err(ExecutorError::JsonError)?;
    let context = seal(ctx.into(), &state.signing_key)?;
    Ok(Hydration { request, context })
}

pub async fn internal_persist(State(state): State<InternalState>, req: Request) -> Response {
    let PersistRequest { context, response, sse } = match read_json(req.into_body()).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    // serde rejects `RawValue` inside `flatten`/`untagged`, so the wire cannot
    // type "exactly one of". Narrow here.
    let upstream = match (response.as_deref(), sse.as_deref()) {
        (Some(json), None) => UpstreamBody::Json(json.get()),
        (None, Some(sse)) => UpstreamBody::Sse(sse),
        _ => {
            let message = "exactly one of `response` or `sse` is required".to_owned();
            return error_response(ExecutorError::InvalidRequest(message));
        }
    };
    let context = match unseal(&context, &state.signing_key) {
        Ok(context) => context,
        Err(error) => return error_response(error),
    };
    let ctx = RequestContext::from(context);
    let stored = match decode_upstream(&ctx, upstream) {
        Ok(payload) => commit(ctx, payload, state.exec_ctx.as_ref()).await,
        Err(error) => Err(error),
    };
    match stored {
        Ok(payload) => axum::Json(payload).into_response(),
        Err(error) => error_response(error),
    }
}

/// Renders an error with the status and envelope core defines.
fn error_response(error: ExecutorError) -> Response {
    let status = error.http_status();
    warn!("backend error ({status}): {error}");
    json(status, error.into_response_body())
}

#[allow(clippy::result_large_err)] // an axum `Response` is the idiomatic error here
async fn read_json<T: DeserializeOwned>(body: Body) -> Result<T, Response> {
    let too_large = br#"{"error":{"type":"invalid_request_error","message":"request body too large"}}"#;
    let bytes = axum::body::to_bytes(body, MAX_BODY_SIZE)
        .await
        .map_err(|_| json(StatusCode::PAYLOAD_TOO_LARGE, too_large.to_vec()))?;
    serde_json::from_slice(&bytes).map_err(|error| error_response(ExecutorError::from(error)))
}

fn json(status: StatusCode, body: Vec<u8>) -> Response {
    Response::builder()
        .status(status)
        .header("Content-Type", "application/json")
        .body(Body::from(body))
        .expect("valid response")
}

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::Router;
use axum::extract::State;
use axum::routing::post;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use agentic_core::config::Config;
use agentic_core::executor::ExecutionContext;
use agentic_core::proxy::ProxyState;
use agentic_core::readiness::llm_readiness_client;
use agentic_server::app::{AppState, ReadinessTracker, WebSocketTracker};
use common::{spawn_gateway, test_config};

/// Bodies the mock upstream received, newest last.
type Recorded = Arc<Mutex<Vec<Value>>>;

/// Mock upstream serving `/v1/chat/completions`, recording every request body.
async fn spawn_mock_upstream() -> (String, Recorded, tokio::task::JoinHandle<()>) {
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(|State(seen): State<Recorded>, body: String| async move {
                if let Ok(value) = serde_json::from_str::<Value>(&body) {
                    seen.lock().expect("recorder mutex").push(value);
                }
                axum::Json(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": "test-model",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "mock reply" },
                        "finish_reason": "stop"
                    }],
                    "usage": { "prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12 }
                }))
            }),
        )
        .with_state(Arc::clone(&recorded));

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock");
    let addr = listener.local_addr().expect("mock addr");
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.expect("serve mock") });
    (format!("http://{addr}"), recorded, handle)
}

/// Build app state backed by a real in-memory database, so the prefix store works.
async fn state_with_storage(config: &Config) -> (AppState, Arc<ExecutionContext>) {
    state_with_compaction(config, None).await
}

async fn state_with_compaction(
    config: &Config,
    compaction_address: Option<String>,
) -> (AppState, Arc<ExecutionContext>) {
    let exec_ctx = Arc::new(ExecutionContext::from_config(config).await.expect("exec ctx"));
    let state = AppState {
        proxy_state: ProxyState::new(config.clone()).expect("proxy state"),
        exec_ctx: Arc::clone(&exec_ctx),
        llm_readiness_client: llm_readiness_client().expect("readiness client"),
        readiness_tracker: ReadinessTracker::default(),
        shutdown_token: CancellationToken::new(),
        websocket_tracker: WebSocketTracker::default(),
        llm_api_base: config.llm_api_base.clone(),
        skip_llm_ready_check: config.skip_llm_ready_check,
        openai_api_key: config.openai_api_key.clone(),
        compaction_address,
    };
    (state, exec_ctx)
}

/// Mock compaction service that folds any history into a single message.
async fn spawn_mock_compactor() -> (String, tokio::task::JoinHandle<()>) {
    let app = Router::new().route(
        "/compact",
        post(|body: String| async move {
            let model = serde_json::from_str::<Value>(&body)
                .ok()
                .and_then(|value| value.get("model").and_then(Value::as_str).map(str::to_owned))
                .unwrap_or_default();
            axum::Json(json!({
                "model": model,
                "messages": [ msg("system", "FOLDED BY COMPACTOR") ]
            }))
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind compactor");
    let addr = listener.local_addr().expect("compactor addr");
    let handle = tokio::spawn(async move { axum::serve(listener, app).await.expect("serve compactor") });
    (format!("http://{addr}/compact"), handle)
}

fn chat_config(upstream: &str) -> Config {
    let mut config = test_config(upstream);
    config.db_url = Some("sqlite://?mode=memory".to_owned());
    config
}

fn msg(role: &str, content: &str) -> Value {
    json!({ "role": role, "content": content })
}

async fn post_chat(url: &str, session: Option<&str>, messages: &[Value]) -> reqwest::Response {
    let mut request = reqwest::Client::new()
        .post(format!("{url}/v1/chat/completions"))
        .json(&json!({ "model": "test-model", "messages": messages }));
    if let Some(session) = session {
        request = request.header("x-session-id", session);
    }
    request.send().await.expect("chat request")
}

fn last_upstream_messages(recorded: &Recorded) -> Vec<Value> {
    recorded
        .lock()
        .expect("recorder mutex")
        .last()
        .and_then(|body| body.get("messages"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// The background write happens after the reply, so give it a moment to land.
async fn await_stored_prefix(exec_ctx: &ExecutionContext, session: &str) -> usize {
    await_stored(exec_ctx, session).await.replaced_count
}

async fn await_stored(
    exec_ctx: &ExecutionContext,
    session: &str,
) -> agentic_core::storage::SessionPrefixData {
    let store = exec_ctx.session_prefix_store();
    for _ in 0..40 {
        if let Ok(Some(prefix)) = store.get(session).await {
            return prefix;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("no prefix stored for {session}");
}

#[tokio::test]
async fn without_a_session_header_the_request_is_forwarded_untouched() {
    let (upstream, recorded, _mock) = spawn_mock_upstream().await;
    let config = chat_config(&upstream);
    let (state, exec_ctx) = state_with_storage(&config).await;
    let (gateway, _gw) = spawn_gateway(state).await;

    let sent = vec![msg("user", "hello")];
    let response = post_chat(&gateway, None, &sent).await;
    assert_eq!(response.status(), 200);

    assert_eq!(last_upstream_messages(&recorded), sent);
    let store = exec_ctx.session_prefix_store();
    assert!(store.get("anything").await.expect("lookup").is_none());
}

#[tokio::test]
async fn a_first_turn_is_forwarded_whole_and_records_the_prefix() {
    let (upstream, recorded, _mock) = spawn_mock_upstream().await;
    let config = chat_config(&upstream);
    let (state, exec_ctx) = state_with_storage(&config).await;
    let (gateway, _gw) = spawn_gateway(state).await;

    let sent = vec![msg("user", "hello")];
    let response = post_chat(&gateway, Some("s-1"), &sent).await;
    assert_eq!(response.status(), 200);

    assert_eq!(last_upstream_messages(&recorded), sent);

    // The turn plus the assistant reply become the new prefix.
    assert_eq!(await_stored_prefix(&exec_ctx, "s-1").await, 2);
}

#[tokio::test]
async fn a_stored_prefix_replaces_the_leading_messages() {
    let (upstream, recorded, _mock) = spawn_mock_upstream().await;
    let config = chat_config(&upstream);
    let (state, exec_ctx) = state_with_storage(&config).await;

    // A distinctive replacement makes the substitution unmistakable.
    exec_ctx
        .session_prefix_store()
        .upsert("s-2", 2, &[msg("system", "FOLDED")])
        .await
        .expect("seed prefix");

    let (gateway, _gw) = spawn_gateway(state).await;

    let sent = vec![msg("user", "one"), msg("assistant", "two"), msg("user", "three")];
    let response = post_chat(&gateway, Some("s-2"), &sent).await;
    assert_eq!(response.status(), 200);

    let upstream_messages = last_upstream_messages(&recorded);
    assert_eq!(upstream_messages.len(), 2, "expected replacement + tail");
    assert_eq!(upstream_messages[0]["content"], "FOLDED");
    assert_eq!(upstream_messages[1]["content"], "three");
}

#[tokio::test]
async fn a_shorter_history_than_stored_is_forwarded_untouched() {
    let (upstream, recorded, _mock) = spawn_mock_upstream().await;
    let config = chat_config(&upstream);
    let (state, exec_ctx) = state_with_storage(&config).await;

    exec_ctx
        .session_prefix_store()
        .upsert("s-3", 5, &[msg("system", "FOLDED")])
        .await
        .expect("seed prefix");

    let (gateway, _gw) = spawn_gateway(state).await;

    let sent = vec![msg("user", "one"), msg("user", "two")];
    let response = post_chat(&gateway, Some("s-3"), &sent).await;
    assert_eq!(response.status(), 200);

    // Substituting would have dropped the whole array, including the new turn.
    assert_eq!(last_upstream_messages(&recorded), sent);
}

#[tokio::test]
async fn a_configured_compactor_shrinks_the_stored_prefix() {
    let (upstream, _recorded, _mock) = spawn_mock_upstream().await;
    let (compactor, _guru) = spawn_mock_compactor().await;
    let config = chat_config(&upstream);
    let (state, exec_ctx) = state_with_compaction(&config, Some(compactor)).await;
    let (gateway, _gw) = spawn_gateway(state).await;

    let sent = vec![msg("user", "one"), msg("assistant", "two"), msg("user", "three")];
    let response = post_chat(&gateway, Some("s-5"), &sent).await;
    assert_eq!(response.status(), 200);

    let stored = await_stored(&exec_ctx, "s-5").await;
    // Three sent plus the reply, folded into one.
    assert_eq!(stored.replaced_count, 4);
    assert_eq!(stored.replacement.len(), 1);
    assert_eq!(stored.replacement[0]["content"], "FOLDED BY COMPACTOR");
}

#[tokio::test]
async fn an_unreachable_compactor_falls_back_to_the_raw_history() {
    let (upstream, _recorded, _mock) = spawn_mock_upstream().await;
    let config = chat_config(&upstream);
    // Nothing listens here; the fold must degrade rather than lose the prefix.
    let (state, exec_ctx) = state_with_compaction(&config, Some("http://127.0.0.1:1/compact".to_owned())).await;
    let (gateway, _gw) = spawn_gateway(state).await;

    let sent = vec![msg("user", "one")];
    assert_eq!(post_chat(&gateway, Some("s-6"), &sent).await.status(), 200);

    let stored = await_stored(&exec_ctx, "s-6").await;
    assert_eq!(stored.replaced_count, 2);
    assert_eq!(stored.replacement.len(), 2, "expected the unmodified history");
}

#[tokio::test]
async fn the_prefix_grows_across_turns() {
    let (upstream, recorded, _mock) = spawn_mock_upstream().await;
    let config = chat_config(&upstream);
    let (state, exec_ctx) = state_with_storage(&config).await;
    let (gateway, _gw) = spawn_gateway(state).await;

    post_chat(&gateway, Some("s-4"), &[msg("user", "one")]).await;
    assert_eq!(await_stored_prefix(&exec_ctx, "s-4").await, 2);

    let turn_two = vec![msg("user", "one"), msg("assistant", "mock reply"), msg("user", "two")];
    let response = post_chat(&gateway, Some("s-4"), &turn_two).await;
    assert_eq!(response.status(), 200);

    // No compactor here, so the stored prefix equals the history and the
    // substituted request matches the original; it must still arrive intact.
    assert_eq!(last_upstream_messages(&recorded).len(), 3);

    for _ in 0..40 {
        if await_stored_prefix(&exec_ctx, "s-4").await == 4 {
            return;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    panic!("prefix did not grow to cover the second turn");
}

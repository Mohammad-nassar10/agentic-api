//! Split execution: hydrate, an external inference call, then persist.

use std::fmt::Write as _;
use std::sync::Arc;

use serde_json::{Value, json};

use agentic_core::executor::request::{RequestContext, SplitContext};
use agentic_core::executor::split::{Hydration, UpstreamBody};
use agentic_core::executor::{ConversationHandler, ExecutionContext, ResponseHandler, rehydrate_conversation, split};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};

async fn exec_ctx() -> ExecutionContext {
    let pool = create_pool_with_schema(Some("sqlite://?mode=memory"))
        .await
        .expect("pool");
    ExecutionContext::new(
        ConversationHandler::new(ConversationStore::new(Arc::clone(&pool))),
        ResponseHandler::new(ResponseStore::new(pool)),
        Arc::new(reqwest::Client::new()),
        "http://localhost:8000".to_owned(),
    )
}

/// Built the way a request actually arrives — through deserialization.
fn request(input: &str, previous: Option<&str>) -> RequestPayload {
    let mut body = json!({"model": "test-model", "input": input, "store": true});
    if let Some(previous) = previous {
        body["previous_response_id"] = json!(previous);
    }
    serde_json::from_value(body).expect("valid request")
}

fn message(status: &str, content: &Value) -> Value {
    json!({"type": "message", "id": "msg_1", "role": "assistant", "status": status, "content": content})
}

/// A complete non-streaming upstream body, as the model backend returns it.
fn upstream_json(body: &str) -> String {
    let text = json!([{"type": "output_text", "text": body, "annotations": []}]);
    json!({
        "id": "resp_upstream", "object": "response", "created_at": 1_700_000_000,
        "model": "test-model", "status": "completed", "output": [message("completed", &text)]
    })
    .to_string()
}

/// The same turn as SSE, the way the caller would have relayed it.
fn upstream_sse(body: &str) -> String {
    let text = json!([{"type": "output_text", "text": body, "annotations": []}]);
    let frames = [
        json!({"type": "response.output_item.added", "output_index": 0, "item": message("in_progress", &json!([]))}),
        json!({"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1", "delta": body}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": message("completed", &text)}),
        json!({"type": "response.completed", "response": {"id": "resp_upstream", "status": "completed"}}),
    ];
    frames.iter().fold(String::new(), |mut sse, frame| {
        writeln!(sse, "data: {frame}\n").expect("write to a String");
        sse
    })
}

async fn hydrate(input: &str, previous: Option<&str>, ctx: &ExecutionContext) -> Hydration {
    split::hydrate(request(input, previous), ctx).await.expect("hydrate")
}

async fn persist(turn: Hydration, upstream: UpstreamBody<'_>, ctx: &ExecutionContext) -> ResponsePayload {
    split::persist(turn.context, upstream, ctx).await.expect("persist")
}

/// The upstream request is raw JSON; parse it to assert on its shape.
fn sent(turn: &Hydration) -> Value {
    serde_json::from_str(turn.request.get()).expect("valid request")
}

/// How many input items the upstream request replays, and their combined text.
fn replayed(turn: &Hydration) -> (usize, String) {
    let items = sent(turn)["input"].as_array().expect("input items").clone();
    (items.len(), items.iter().map(ToString::to_string).collect())
}

/// `InputItem` and `ResponsesTool` are not `PartialEq`; compare their wire form.
fn json_of<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("serializable")
}

#[tokio::test]
async fn a_second_turn_replays_the_stored_history() {
    let ctx = exec_ctx().await;

    let turn = hydrate("What is 2+2?", None, &ctx).await;
    assert_eq!(replayed(&turn).0, 1);
    assert!(
        sent(&turn).get("previous_response_id").is_none(),
        "upstream is stateless"
    );

    let first = persist(turn, UpstreamBody::Json(&upstream_json("4")), &ctx).await;
    assert!(first.id.starts_with("resp_"));
    assert_ne!(
        first.id, "resp_upstream",
        "the envelope carries our id, not the model's"
    );

    let turn = hydrate("What did I ask?", Some(&first.id), &ctx).await;
    let (count, text) = replayed(&turn);
    assert_eq!(count, 3, "prior user + assistant turns, then the new input");
    assert!(text.contains("What is 2+2?") && text.contains('4'));

    let second = persist(turn, UpstreamBody::Json(&upstream_json("2+2")), &ctx).await;
    assert_eq!(second.previous_response_id.as_deref(), Some(first.id.as_str()));
}

/// Every way a turn can fail to be stored, and the status the caller sees.
#[tokio::test]
async fn a_turn_that_cannot_be_stored_is_refused() {
    let ctx = exec_ctx().await;

    let unknown = split::hydrate(request("hi", Some("resp_missing")), &ctx).await;
    assert_eq!(unknown.expect_err("unknown id").http_status().as_u16(), 404);

    let mut in_progress: Value = serde_json::from_str(&upstream_json("partial")).expect("json");
    in_progress["status"] = json!("in_progress");
    let turn = hydrate("hi", None, &ctx).await;
    let error = split::persist(turn.context, UpstreamBody::Json(&in_progress.to_string()), &ctx).await;
    assert_eq!(error.expect_err("never stored").http_status().as_u16(), 400);

    // A relay that died mid-stream: `finish_stream` would call this complete.
    let cut_short = r#"data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"4"}"#;
    let turn = hydrate("hi", None, &ctx).await;
    let error = split::persist(turn.context, UpstreamBody::Sse(cut_short), &ctx).await;
    assert_eq!(error.expect_err("no terminal event").http_status().as_u16(), 400);
}

#[test]
fn the_boundary_check_names_what_cannot_be_split() {
    let gateway_tool: RequestPayload = serde_json::from_value(json!({
        "model": "test-model", "input": "hi", "store": true, "tools": [{"type": "web_search_preview"}]
    }))
    .expect("valid request");
    let error = split::ensure_splittable(&gateway_tool).expect_err("the loop needs a caller");
    assert!(error.to_string().contains("tools"), "got: {error}");

    let mut conversational = request("hi", None);
    conversational.conversation_id = Some("conv_1".into());
    let error = split::ensure_splittable(&conversational).expect_err("its version cannot cross");
    assert!(error.to_string().contains("conversation_id"), "got: {error}");

    let mut streaming = request("hi", None);
    streaming.stream = true;
    split::ensure_splittable(&streaming).expect("the caller relays the frames, then replays them");
    split::ensure_splittable(&request("hi", None)).expect("a plain turn is splittable");
}

/// The wire form drops what it can rebuild, and the rebuild has to agree with
/// what hydration produced — persist stores from it.
#[tokio::test]
async fn the_wire_context_round_trips_into_an_equal_context() {
    let ctx = exec_ctx().await;
    let live = rehydrate_conversation(request("What is 2+2?", None), &ctx)
        .await
        .expect("rehydrate");
    let (id, items, tools) = (
        live.response_id.clone(),
        json_of(&live.new_input_items),
        json_of(&live.enriched_request.tools),
    );

    let wire = serde_json::to_string(&SplitContext::from(live)).expect("serialize");
    assert!(!wire.contains("enriched_request"), "already in flight as the request");
    assert!(
        !wire.contains("conversation_version"),
        "conversation mode does not split"
    );

    let back = RequestContext::from(serde_json::from_str::<SplitContext>(&wire).expect("deserialize"));
    assert_eq!(back.response_id, id);
    assert_eq!(json_of(&back.new_input_items), items, "derived items match");
    assert_eq!(json_of(&back.enriched_request.tools), tools, "resolved tools survive");
    assert!(
        back.enriched_request.previous_response_id.is_none(),
        "upstream stays stateless"
    );
    assert!(
        back.conversation_version.is_none(),
        "never resumed with a stale version"
    );
}

#[tokio::test]
async fn a_streamed_turn_persists_from_the_relayed_frames() {
    let ctx = exec_ctx().await;
    let mut streaming = request("What is 2+2?", None);
    streaming.stream = true;

    let turn = split::hydrate(streaming, &ctx).await.expect("hydrate");
    assert_eq!(
        sent(&turn)["stream"],
        json!(true),
        "the client's flag reaches the model"
    );

    let stored = persist(turn, UpstreamBody::Sse(&upstream_sse("4")), &ctx).await;
    assert_ne!(stored.id, "resp_upstream");

    let next = hydrate("What did I ask?", Some(&stored.id), &ctx).await;
    assert_eq!(replayed(&next).0, 3, "the streamed turn is continuable");
}

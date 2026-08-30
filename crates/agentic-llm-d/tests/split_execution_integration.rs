//! Split execution at the library level: what the handlers compose, and the
//! cases HTTP cannot reach. The round trip over a socket is in `backend_mode_test`.
#![allow(clippy::result_large_err)] // `ExecutorError` is core's; boxing it is not ours to decide

use std::sync::Arc;

use agentic_core::executor::request::RequestContext;
use agentic_core::executor::{
    ConversationHandler, ExecutionContext, ExecutorError, ExecutorResult, ResponseHandler, UpstreamBody, commit,
    decode_upstream, rehydrate_conversation, upstream_request,
};
use agentic_core::storage::{ConversationStore, ResponseStore, create_pool_with_schema};
use agentic_core::types::request_response::{RequestPayload, ResponsePayload};
use agentic_llm_d::context::{Hydration, SplitContext, ensure_splittable, seal, unseal};
use serde_json::value::RawValue;
use serde_json::{Value, json};

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

fn answer(text: &str) -> Value {
    json!([{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}]}])
}

/// A complete upstream body, as the model backend returns it.
fn upstream_json(text: &str) -> String {
    json!({"id": "resp_upstream", "object": "response", "created_at": 1_700_000_000,
           "model": "test-model", "status": "completed", "output": answer(text)})
    .to_string()
}

/// The same turn as SSE, the way a streaming caller would have relayed it.
fn upstream_sse(text: &str) -> String {
    [
        json!({"type": "response.output_item.added", "output_index": 0,
               "item": {"type": "message", "id": "msg_1", "role": "assistant", "status": "in_progress", "content": []}}),
        json!({"type": "response.output_text.delta", "output_index": 0, "item_id": "msg_1", "delta": text}),
        json!({"type": "response.output_item.done", "output_index": 0, "item": answer(text)[0]}),
        json!({"type": "response.completed", "response": {"id": "resp_upstream", "status": "completed"}}),
    ]
    .iter()
    .map(|frame| format!("data: {frame}\n\n"))
    .collect::<Vec<_>>()
    .concat()
}

const KEY: &[u8] = b"test-signing-key";

/// What the `/internal/hydrate` handler composes.
async fn hydrate(request: RequestPayload, ctx: &ExecutionContext) -> ExecutorResult<Hydration> {
    ensure_splittable(&request)?;
    let live = rehydrate_conversation(request, ctx).await?;
    let stream = live.original_request.stream;
    let body = RawValue::from_string(upstream_request(&live, stream)?).map_err(ExecutorError::JsonError)?;
    Ok(Hydration {
        request: body,
        context: seal(live.into(), KEY)?,
    })
}

/// What the `/internal/persist` handler composes.
async fn persist(
    context: String,
    upstream: UpstreamBody<'_>,
    ctx: &ExecutionContext,
) -> ExecutorResult<ResponsePayload> {
    let live = RequestContext::from(unseal(&context, KEY)?);
    let payload = decode_upstream(&live, upstream)?;
    commit(live, payload, ctx).await
}

async fn turn(input: &str, previous: Option<&str>, ctx: &ExecutionContext) -> Hydration {
    hydrate(request(input, previous), ctx).await.expect("hydrate")
}

/// How many input items the upstream request replays, and their combined text.
fn replayed(turn: &Hydration) -> (usize, String) {
    let sent: Value = serde_json::from_str(turn.request.get()).expect("valid request");
    let items = sent["input"].as_array().expect("input items").clone();
    (items.len(), items.iter().map(ToString::to_string).collect())
}

fn status_of(error: &ExecutorError) -> u16 {
    error.http_status().as_u16()
}

#[tokio::test]
async fn a_streamed_turn_persists_from_the_relayed_frames() {
    let ctx = exec_ctx().await;
    let mut streaming = request("What is 2+2?", None);
    streaming.stream = true;
    let streamed = hydrate(streaming, &ctx).await.expect("hydrate");

    let stored = persist(streamed.context, UpstreamBody::Sse(&upstream_sse("4")), &ctx)
        .await
        .expect("persist from SSE");
    let next = turn("What did I ask?", Some(&stored.id), &ctx).await;
    assert_eq!(replayed(&next).0, 3, "the streamed turn is continuable");
}

/// Every way a turn is refused, and the status the caller sees.
#[tokio::test]
async fn a_turn_that_cannot_be_stored_is_refused() {
    let ctx = exec_ctx().await;

    let unknown = hydrate(request("hi", Some("resp_missing")), &ctx).await;
    assert_eq!(status_of(&unknown.expect_err("unknown id")), 404);

    // The in-process parser defaults a missing status to `completed` and drops
    // items it cannot read; a caller-supplied body gets neither. An unrecognized
    // item *type* is still fine — `OutputItem` keeps a catch-all.
    let mut in_progress: Value = serde_json::from_str(&upstream_json("partial")).expect("json");
    in_progress["status"] = json!("in_progress");
    for body in [
        in_progress.to_string(),
        r#"{"id":"resp_upstream"}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed"}"#.to_owned(),
        r#"{"id":"resp_upstream","status":"completed","output":[123]}"#.to_owned(),
    ] {
        let refused = persist(turn("hi", None, &ctx).await.context, UpstreamBody::Json(&body), &ctx).await;
        assert_eq!(status_of(&refused.expect_err("not storable")), 400, "accepted: {body}");
    }

    // A relay that died mid-stream: `finish_stream` would call this complete.
    let cut_short = r#"data: {"type":"response.output_text.delta","item_id":"msg_1","delta":"4"}"#;
    let refused = persist(turn("hi", None, &ctx).await.context, UpstreamBody::Sse(cut_short), &ctx).await;
    assert_eq!(status_of(&refused.expect_err("no terminal event")), 400);

    let opened = unseal(&turn("hi", None, &ctx).await.context, KEY).expect("unseal");
    let no_id = seal(
        SplitContext {
            response_id: String::new(),
            ..opened
        },
        KEY,
    )
    .expect("seal");
    let refused = persist(no_id, UpstreamBody::Json(&upstream_json("4")), &ctx).await;
    assert_eq!(status_of(&refused.expect_err("no reserved id")), 400);
}

/// Two turns that are not written but must not read as errors: a model that
/// failed, and a retry of one already stored.
#[tokio::test]
async fn a_turn_that_is_not_written_still_returns() {
    let ctx = exec_ctx().await;

    let mut failed: Value = serde_json::from_str(&upstream_json("")).expect("json");
    failed["status"] = json!("failed");
    failed["output"] = json!([]);
    let attempt = turn("hi", None, &ctx).await;
    let id = unseal(&attempt.context, KEY).expect("unseal").response_id;
    let payload = persist(attempt.context, UpstreamBody::Json(&failed.to_string()), &ctx)
        .await
        .expect("a failed turn is not a boundary error");
    assert_eq!(payload.status, "error", "`failed` normalizes to the error status");
    let orphan = hydrate(request("and then?", Some(&id)), &ctx).await;
    assert_eq!(status_of(&orphan.expect_err("never stored")), 404);

    let stored = turn("What is 2+2?", None, &ctx).await;
    let context = stored.context.clone();
    let first = persist(stored.context, UpstreamBody::Json(&upstream_json("4")), &ctx)
        .await
        .expect("persist");
    let retry = persist(context, UpstreamBody::Json(&upstream_json("4")), &ctx)
        .await
        .expect("a retry is not a failure");
    assert_eq!(retry.id, first.id);
    assert_eq!(replayed(&turn("and?", Some(&first.id), &ctx).await).0, 3, "stored once");
}

#[test]
fn the_boundary_check_names_what_cannot_be_split() {
    let gateway_tool: RequestPayload = serde_json::from_value(json!({
        "model": "test-model", "input": "hi", "store": true, "tools": [{"type": "web_search_preview"}]
    }))
    .expect("valid request");
    let error = ensure_splittable(&gateway_tool).expect_err("the loop needs a caller");
    assert!(error.to_string().contains("tools"), "got: {error}");

    let mut conversational = request("hi", None);
    conversational.conversation_id = Some("conv_1".into());
    let error = ensure_splittable(&conversational).expect_err("its version cannot cross");
    assert!(error.to_string().contains("conversation_id"), "got: {error}");

    let mut streaming = request("hi", None);
    streaming.stream = true;
    ensure_splittable(&streaming).expect("the caller relays the frames, then replays them");
    ensure_splittable(&request("hi", None)).expect("a plain turn is splittable");
}

/// The wire form drops what it can rebuild, and the rebuild has to agree with
/// what hydration produced — persist stores from it.
#[tokio::test]
async fn the_wire_context_round_trips_into_an_equal_context() {
    let ctx = exec_ctx().await;
    let live = rehydrate_conversation(request("What is 2+2?", None), &ctx)
        .await
        .expect("rehydrate");
    let (id, items) = (live.response_id.clone(), live.new_input_items.len());

    let wire = serde_json::to_string(&SplitContext::from(live)).expect("serialize");
    assert!(!wire.contains("enriched_request"), "already in flight as the request");
    assert!(
        !wire.contains("conversation_version"),
        "conversation mode does not split"
    );

    let back = RequestContext::from(serde_json::from_str::<SplitContext>(&wire).expect("deserialize"));
    assert_eq!(back.response_id, id);
    assert_eq!(back.new_input_items.len(), items, "derived items match");
    assert!(
        back.enriched_request.previous_response_id.is_none(),
        "upstream stays stateless"
    );
    assert!(
        back.conversation_version.is_none(),
        "never resumed with a stale version"
    );
}

/// A context `hydrate` did not issue must not be usable: without this a caller
/// could skip hydration and write turns under any id it chose.
#[tokio::test]
async fn a_context_this_service_did_not_seal_is_rejected() {
    let ctx = exec_ctx().await;
    let sealed = turn("hi", None, &ctx).await.context;

    for forged in [
        seal(unseal(&sealed, KEY).expect("unseal"), b"a-different-key").expect("seal"),
        format!("{sealed}tampered"),
    ] {
        let refused = persist(forged, UpstreamBody::Json(&upstream_json("4")), &ctx).await;
        assert_eq!(status_of(&refused.expect_err("not ours")), 400);
    }

    // The one we did issue still works.
    persist(sealed, UpstreamBody::Json(&upstream_json("4")), &ctx)
        .await
        .expect("our own context");
}

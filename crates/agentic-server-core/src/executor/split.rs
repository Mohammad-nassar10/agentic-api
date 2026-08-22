//! Split execution — hydration and persistence as separately callable steps,
//! for an orchestrator (e.g. the llm-d coordinator) that runs inference itself.
//!
//! [`hydrate`] returns the stateless upstream request plus an opaque
//! [`HydrationContext`]; the orchestrator calls the model, then passes the
//! context and the response to [`persist`].
//!
//! This module is a re-composition, not a reimplementation: both halves call the
//! same steps the in-process flow uses — [`rehydrate_conversation`], then
//! `payload_from_upstream_body` and `persist_if_needed`, the latter two shared
//! with `fetch_blocking_payload`. What it adds is [`HydrationContext`] — the in-process
//! flow threads a [`RequestContext`] between those steps, which is not
//! serializable and so cannot cross the process boundary that now sits between
//! them. It carries the minimum needed to rebuild one.
//!
//! Splitting the flow also gives up in-process-only behavior (streaming, gateway
//! tool loops, compaction), so [`hydrate`] rejects requests that need it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::persist::persist_if_needed;
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::executor::upstream::payload_from_upstream_body;
use crate::types::event::ResponseStatus;
use crate::types::io::{ResponsesInput, ToolChoice};
use crate::types::request_response::{RequestPayload, ResponsePayload};
use crate::types::tools::ResponsesTool;
use crate::utils::common::{serialize_to_string, serialize_to_value};

/// State captured at hydration time that [`persist`] needs to store the turn.
/// Serializable so it can travel to the orchestrator; echo it back unchanged.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HydrationContext {
    /// Response id reserved for this turn (`resp_` prefix).
    pub response_id: String,
    /// The client's request as received, continuation ids intact.
    pub original_request: RequestPayload,
    pub conversation_id: Option<String>,
    pub effective_tools: Option<Vec<ResponsesTool>>,
    pub effective_tool_choice: Option<ToolChoice>,
}

impl HydrationContext {
    /// Rebuilds the [`RequestContext`] the persistence path expects.
    ///
    /// `new_input_items` is derived rather than carried: only compaction and the
    /// gateway tool loop ever diverge it from the request's own input, and
    /// [`ensure_supported`] rejects both.
    fn into_request_context(self) -> RequestContext {
        let new_input_items = Vec::from(&self.original_request.input);
        let mut enriched_request = self.original_request.clone();
        enriched_request.previous_response_id = None;
        enriched_request.input = ResponsesInput::Items(new_input_items.clone());
        enriched_request.tools = self.effective_tools;
        enriched_request.tool_choice = self.effective_tool_choice;
        RequestContext {
            original_request: self.original_request,
            enriched_request,
            new_input_items,
            response_id: self.response_id,
            conversation_id: self.conversation_id,
            conversation_version: None,
        }
    }
}

/// Result of [`hydrate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hydration {
    /// Upstream request body: history inlined, no continuation or storage fields.
    pub request: Value,
    /// Echo back to [`persist`] unchanged.
    pub context: HydrationContext,
}

/// Step 1 — rehydrate history and build the upstream request.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for requests needing in-process-only
/// behavior (streaming, `conversation_id`, gateway tools, compaction), not-found
/// for an unknown `previous_response_id`, or a storage error.
pub async fn hydrate(request: RequestPayload, exec_ctx: &ExecutionContext) -> ExecutorResult<Hydration> {
    ensure_supported(&request)?;
    let ctx = rehydrate_conversation(request, exec_ctx).await?;
    let upstream_request = ctx.enriched_request.to_upstream_request(false)?;
    let request_value = serialize_to_value(&upstream_request).map_err(ExecutorError::JsonError)?;
    Ok(Hydration {
        request: request_value,
        context: HydrationContext {
            response_id: ctx.response_id,
            original_request: ctx.original_request,
            conversation_id: ctx.conversation_id,
            effective_tools: ctx.enriched_request.tools,
            effective_tool_choice: ctx.enriched_request.tool_choice,
        },
    })
}

/// Step 2 — persist the turn and build the final envelope.
///
/// `upstream_response` is the complete non-streaming upstream body. Requests
/// that opted out of storage are returned without being stored.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] if the response is not in a terminal state,
/// a parse error if it is not a valid response body, or a storage error.
pub async fn persist(
    context: HydrationContext,
    upstream_response: &Value,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<ResponsePayload> {
    let ctx = context.into_request_context();
    let body = serialize_to_string(upstream_response).map_err(ExecutorError::JsonError)?;
    let payload = payload_from_upstream_body(&ctx, &body)?;

    // The in-process flow silently skips non-terminal statuses; here that would
    // return an envelope whose id can never be continued, so reject it.
    if !matches!(
        payload.status.parse::<ResponseStatus>().unwrap_or_default(),
        ResponseStatus::Completed | ResponseStatus::Incomplete
    ) {
        return Err(ExecutorError::InvalidRequest(format!(
            "upstream response status '{}' cannot be persisted",
            payload.status
        )));
    }

    persist_if_needed(
        payload.clone(),
        ctx,
        exec_ctx.conv_handler.clone(),
        exec_ctx.resp_handler.clone(),
    )
    .await?;
    Ok(payload)
}

/// Rejects requests whose behavior a hydrate/persist split cannot reproduce.
fn ensure_supported(request: &RequestPayload) -> ExecutorResult<()> {
    if request.stream {
        return Err(ExecutorError::InvalidRequest(
            "stream is not supported for split execution".into(),
        ));
    }
    if let Some(feature) = request.in_process_feature() {
        return Err(ExecutorError::InvalidRequest(format!(
            "{feature} is not supported for split execution"
        )));
    }
    Ok(())
}

//! Split execution — hydration and persistence as separately callable steps,
//! for an orchestrator (e.g. the llm-d coordinator) that runs inference itself.
//!
//! [`hydrate`] returns the stateless upstream request plus the [`RequestContext`]
//! for the turn; the orchestrator calls the model, then passes the context and
//! the response to [`persist`].
//!
//! Both halves call the same steps the in-process flow uses rather than
//! duplicating them — [`rehydrate_conversation`], then
//! `payload_from_upstream`, `persist_if_needed`. The context is the same type
//! either way; it serializes through a reduced wire form so only the fields that
//! mean anything off-process make the trip.

use serde::{Deserialize, Serialize};
use serde_json::value::RawValue;

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::persist::persist_if_needed;
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext, SplitContext};
pub use crate::executor::upstream::UpstreamBody;
use crate::executor::upstream::{payload_from_upstream, upstream_request_json};
use crate::types::event::ResponseStatus;
use crate::types::request_response::{RequestPayload, ResponsePayload};

/// Result of [`hydrate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hydration {
    /// Upstream request body: history inlined, no continuation or storage fields.
    /// Raw JSON — the caller forwards it without interpreting it.
    pub request: Box<RawValue>,
    /// Echo back to [`persist`] unchanged.
    pub context: SplitContext,
}

/// Rehydrates history and builds the upstream request.
///
/// Rejects requests that cannot be split; see [`ensure_splittable`], which is
/// public so a caller can pre-validate for a cleaner error.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for a request that cannot be split,
/// not-found for an unknown `previous_response_id`, or a storage error.
pub async fn hydrate(request: RequestPayload, exec_ctx: &ExecutionContext) -> ExecutorResult<Hydration> {
    ensure_splittable(&request)?;
    let ctx = rehydrate_conversation(request, exec_ctx).await?;
    let stream = ctx.original_request.stream;
    let request = RawValue::from_string(upstream_request_json(&ctx, stream)?).map_err(ExecutorError::JsonError)?;
    Ok(Hydration {
        request,
        context: ctx.into(),
    })
}

/// Persists the turn and builds the final envelope.
///
/// For a stream the caller has already relayed the frames, so nothing is
/// re-emitted here. Stored when the request set `store` or continues a chain —
/// `store: false` alone skips storage, but not alongside `previous_response_id`.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for a non-terminal response or a cut-short
/// stream, a parse error for an invalid body, or a storage error.
pub async fn persist(
    context: SplitContext,
    upstream: UpstreamBody<'_>,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<ResponsePayload> {
    let ctx = RequestContext::from(context);
    let payload = payload_from_upstream(&ctx, upstream)?;

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

/// Rejects requests that cannot cross a process boundary — each needs state the
/// in-process flow keeps between steps. Not a limit of hydration itself; callers
/// running the whole turn in-process should not call this.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] naming the feature that cannot be split.
pub fn ensure_splittable(request: &RequestPayload) -> ExecutorResult<()> {
    if let Some(feature) = request.in_process_feature() {
        return Err(ExecutorError::InvalidRequest(format!(
            "{feature} is not supported for split execution"
        )));
    }
    Ok(())
}

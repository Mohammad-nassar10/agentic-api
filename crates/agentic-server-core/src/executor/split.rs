//! Split execution — hydration, tool rounds and persistence as separately
//! callable steps, for an orchestrator (e.g. the llm-d coordinator) that runs
//! inference itself.
//!
//! [`hydrate`] returns the stateless upstream request plus an opaque
//! [`HydrationContext`]; the orchestrator calls the model, then passes the
//! context and the response to [`persist`]. When the request declares
//! gateway-owned tools it calls [`tool_round`] in between, once per model
//! response, until it reports [`ToolRound::Done`].
//!
//! This module is a re-composition, not a reimplementation: every half calls the
//! same steps the in-process flow uses — [`rehydrate_conversation`], then
//! `payload_from_upstream_body` and `persist_if_needed`, the latter two shared
//! with `fetch_blocking_payload`; the tool round shares `execute_output_calls`,
//! `classify_round` and the append helpers with `engine::run_gateway_tool_loop`.
//! What it adds is [`HydrationContext`] — the in-process flow threads a
//! [`RequestContext`] between those steps, which is not serializable and so
//! cannot cross the process boundary that now sits between them. It carries the
//! minimum needed to rebuild one.
//!
//! Splitting the flow still gives up in-process-only behavior (streaming,
//! conversation mode, compaction), so [`hydrate`] rejects requests that need it.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::executor::engine::{MAX_GATEWAY_TOOL_ROUNDS, accumulate_usage};
use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::executor::gateway::{
    GatewayCallResult, LoopDecision, append_gateway_calls_to_new_input, append_output_items_to_input,
    append_tool_outputs, classify_round, execute_output_calls, has_client_owned_calls, public_output_items,
};
use crate::executor::persist::persist_if_needed;
use crate::executor::rehydrate::rehydrate_conversation;
use crate::executor::request::{ExecutionContext, RequestContext};
use crate::executor::upstream::payload_from_upstream_body;
use crate::tool::{ToolRegistry, mcp};
use crate::types::event::ResponseStatus;
use crate::types::io::{InputItem, OutputItem, ResponseUsage, ResponsesInput, ToolChoice};
use crate::types::request_response::{IncompleteDetails, RequestPayload, ResponsePayload};
use crate::types::tools::ResponsesTool;
use crate::utils::common::{deserialize_from_value, serialize_to_string, serialize_to_value};

/// Tool-loop state accumulated across rounds. Absent for a turn with no
/// gateway-owned tools, so a plain hydrate/persist context is unchanged.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolLoopState {
    /// Rounds already executed.
    pub round: usize,
    /// This turn's items to store: the client's input, then each round's
    /// gateway calls and their outputs.
    pub new_input_items: Vec<InputItem>,
    /// Client-visible items from rounds before the last (`mcp_list_tools`,
    /// `mcp_call`, `web_search_call`). The envelope is built from the final
    /// response, which does not carry them.
    pub prior_output: Vec<OutputItem>,
    /// Usage from rounds before the last; the final round's comes from the
    /// response [`persist`] is given.
    pub usage: Option<ResponseUsage>,
    /// Set when the round budget ran out, making the stored turn `incomplete`.
    pub incomplete_reason: Option<String>,
}

/// State captured at hydration time that later steps need to store the turn.
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
    /// Present only once a tool round has run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_state: Option<ToolLoopState>,
}

impl HydrationContext {
    /// Rebuilds the [`RequestContext`] the later steps expect.
    ///
    /// Without a tool loop `new_input_items` is derived rather than carried:
    /// only compaction and the gateway tool loop ever diverge it from the
    /// request's own input, and [`ensure_supported`] rejects the former.
    fn into_request_context(self) -> RequestContext {
        let new_input_items = match &self.tool_state {
            Some(state) => state.new_input_items.clone(),
            None => Vec::from(&self.original_request.input),
        };
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
    /// Echo back to [`tool_round`] and [`persist`] unchanged.
    pub context: HydrationContext,
    /// Whether this turn needs [`tool_round`] between inference and [`persist`].
    pub gateway_tools: bool,
}

/// Result of [`tool_round`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum ToolRound {
    /// Gateway tools ran: send `request` upstream and call [`tool_round`] again
    /// with the new response.
    Continue { request: Value, context: HydrationContext },
    /// No gateway work remains; the response just handed in is the final one —
    /// pass it to [`persist`] with this context.
    Done { context: HydrationContext },
}

/// Step 1 — rehydrate history and build the upstream request.
///
/// `tool_loop` declares that the caller implements [`tool_round`]; without it a
/// request carrying gateway-owned tools is rejected rather than answered with
/// its tools silently unexecuted.
///
/// # Errors
/// [`ExecutorError::InvalidRequest`] for requests needing in-process-only
/// behavior (streaming, `conversation_id`, compaction, and gateway tools unless
/// `tool_loop`), not-found for an unknown `previous_response_id`, or a storage
/// error.
pub async fn hydrate(
    request: RequestPayload,
    exec_ctx: &ExecutionContext,
    tool_loop: bool,
) -> ExecutorResult<Hydration> {
    ensure_supported(&request, tool_loop)?;
    let mut ctx = rehydrate_conversation(request, exec_ctx).await?;
    let gateway_tools = ctx.enriched_request.in_process_feature() == Some(RequestPayload::GATEWAY_TOOLS_FEATURE);
    // Discovery fills in each MCP declaration's tool list in place, and
    // `to_upstream_request` renders the model-visible functions from it — so it
    // has to run first, exactly as the in-process loop builds its registry
    // before the first inference call.
    if gateway_tools && let Some(tools) = ctx.enriched_request.tools.as_mut() {
        ToolRegistry::build_with_handlers(tools, &mut exec_ctx.gateway_executors.request_scoped()).await?;
    }
    let upstream_request = ctx.enriched_request.to_upstream_request(false)?;
    let request_value = serialize_to_value(&upstream_request).map_err(ExecutorError::JsonError)?;
    Ok(Hydration {
        request: request_value,
        gateway_tools,
        context: HydrationContext {
            response_id: ctx.response_id,
            original_request: ctx.original_request,
            conversation_id: ctx.conversation_id,
            effective_tools: ctx.enriched_request.tools,
            effective_tool_choice: ctx.enriched_request.tool_choice,
            tool_state: None,
        },
    })
}

/// Step 2 (only when [`Hydration::gateway_tools`]) — run one round of the
/// gateway tool loop.
///
/// `upstream_request` is the body the caller just sent; `upstream_response` is
/// what came back. Gateway-owned calls are executed here and their outputs
/// appended, so the returned request is the next one to send.
///
/// # Errors
/// A parse error if either body is malformed, or a tool-configuration error.
pub async fn tool_round(
    context: HydrationContext,
    upstream_request: &Value,
    upstream_response: &Value,
    exec_ctx: &ExecutionContext,
) -> ExecutorResult<ToolRound> {
    let mut state = context.tool_state.clone().unwrap_or_default();
    let mut ctx = context.into_request_context();
    // Continue from the conversation as it was actually sent upstream this
    // round; the rebuilt context only carries this turn's own items.
    ctx.enriched_request.input = upstream_input(upstream_request)?;

    let mut executors = exec_ctx.gateway_executors.request_scoped();
    let registry = match ctx.enriched_request.tools.as_mut() {
        Some(tools) => ToolRegistry::build_with_handlers(tools, &mut executors).await?,
        None => ToolRegistry::default(),
    };
    if state.round == 0 {
        state.prior_output.extend(
            registry
                .mcp_list_tools_items()
                .iter()
                .map(mcp::handler::list_tools_output_item),
        );
    }

    let body = serialize_to_string(upstream_response).map_err(ExecutorError::JsonError)?;
    let mut payload = payload_from_upstream_body(&ctx, &body)?;
    registry.restore_final_payload_output(&mut payload.output);
    let round_usage = payload.usage.take();
    let current_output = payload.output;

    let has_client_owned = has_client_owned_calls(&current_output, &registry);
    let gateway_results = execute_output_calls(&current_output, &registry).await?;
    let public_output = public_output_items(&current_output, &registry, &gateway_results);

    let mut another_round = false;
    match classify_round(has_client_owned, &gateway_results, state.round, MAX_GATEWAY_TOOL_ROUNDS) {
        // This turn is final; `persist` builds the envelope from the same
        // response, so nothing from it is accumulated here.
        LoopDecision::Done => {}
        // Handed back to the client, or out of rounds: the calls and any
        // outputs are still recorded so the stored turn has no dangling call.
        LoopDecision::RequiresClientAction => record_round(&mut ctx, &current_output, &registry, gateway_results),
        LoopDecision::Incomplete(reason) => {
            record_round(&mut ctx, &current_output, &registry, gateway_results);
            state.incomplete_reason = Some(reason);
        }
        LoopDecision::Continue => {
            ctx.enriched_request.tool_choice = Some(ToolChoice::Auto);
            append_output_items_to_input(&mut ctx.enriched_request.input, &current_output);
            record_round(&mut ctx, &current_output, &registry, gateway_results);
            // This response is not the one `persist` sees, so its output and
            // usage would otherwise be lost.
            state.prior_output.extend(public_output);
            accumulate_usage(&mut state.usage, round_usage);
            another_round = true;
        }
    }

    state.new_input_items = ctx.new_input_items.clone();
    state.round += 1;
    let next_request = if another_round {
        let upstream = ctx.enriched_request.to_upstream_request(false)?;
        Some(serialize_to_value(&upstream).map_err(ExecutorError::JsonError)?)
    } else {
        None
    };

    let context = HydrationContext {
        response_id: ctx.response_id,
        original_request: ctx.original_request,
        conversation_id: ctx.conversation_id,
        effective_tools: ctx.enriched_request.tools,
        effective_tool_choice: ctx.enriched_request.tool_choice,
        tool_state: Some(state),
    };
    Ok(match next_request {
        Some(request) => ToolRound::Continue { request, context },
        None => ToolRound::Done { context },
    })
}

/// Step 3 — persist the turn and build the final envelope.
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
    let state = context.tool_state.clone().unwrap_or_default();
    let ctx = context.into_request_context();
    let body = serialize_to_string(upstream_response).map_err(ExecutorError::JsonError)?;
    let mut payload = payload_from_upstream_body(&ctx, &body)?;

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

    apply_tool_loop_state(&mut payload, state);

    persist_if_needed(
        payload.clone(),
        ctx,
        exec_ctx.conv_handler.clone(),
        exec_ctx.resp_handler.clone(),
    )
    .await?;
    Ok(payload)
}

/// Folds earlier rounds into the final envelope: their client-visible items lead
/// the output, their usage is summed in, and an exhausted round budget makes the
/// turn `incomplete`.
fn apply_tool_loop_state(payload: &mut ResponsePayload, state: ToolLoopState) {
    if !state.prior_output.is_empty() {
        let mut output = state.prior_output;
        output.append(&mut payload.output);
        payload.output = output;
    }
    accumulate_usage(&mut payload.usage, state.usage);
    if let Some(reason) = state.incomplete_reason {
        "incomplete".clone_into(&mut payload.status);
        payload.incomplete_details = Some(IncompleteDetails { reason: Some(reason) });
    }
}

/// Records a round's gateway calls and their outputs as items to store.
fn record_round(
    ctx: &mut RequestContext,
    output_items: &[OutputItem],
    registry: &ToolRegistry,
    results: Vec<GatewayCallResult>,
) {
    append_gateway_calls_to_new_input(ctx, output_items, registry);
    append_tool_outputs(ctx, results.into_iter().map(|result| result.input_item).collect());
}

/// Reads the `input` of a request body that was sent upstream.
fn upstream_input(upstream_request: &Value) -> ExecutorResult<ResponsesInput> {
    let input = upstream_request
        .get("input")
        .cloned()
        .ok_or_else(|| ExecutorError::InvalidRequest("upstream request has no 'input'".into()))?;
    deserialize_from_value(input).map_err(ExecutorError::JsonError)
}

/// Rejects requests whose behavior a split flow cannot reproduce.
fn ensure_supported(request: &RequestPayload, tool_loop: bool) -> ExecutorResult<()> {
    if request.stream {
        return Err(ExecutorError::InvalidRequest(
            "stream is not supported for split execution".into(),
        ));
    }
    // Gateway tools are checked last, so reaching them means nothing else
    // matched and a caller running the tool loop can proceed.
    if let Some(feature) = request.in_process_feature() {
        if !(tool_loop && feature == RequestPayload::GATEWAY_TOOLS_FEATURE) {
            return Err(ExecutorError::InvalidRequest(format!(
                "{feature} is not supported for split execution"
            )));
        }
    }
    Ok(())
}

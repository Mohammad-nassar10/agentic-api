//! Cluster-internal split-execution endpoints for an orchestrator that runs
//! inference itself. They trust their caller: restrict them at the network
//! layer, never expose them on a public route.

use axum::extract::{Query, Request, State};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::Value;

use agentic_core::executor::split;
use agentic_core::executor::split::HydrationContext;
use agentic_core::types::request_response::RequestPayload;

use super::super::common::{executor_error_response, read_json};
use crate::app::AppState;

/// Query of `POST /internal/hydrate`.
#[derive(Debug, Default, Deserialize)]
pub struct HydrateQuery {
    /// Set by a caller that runs the tool loop via `POST /internal/tools`.
    /// Without it a request declaring gateway-owned tools is rejected rather
    /// than answered with those tools silently unexecuted.
    #[serde(default)]
    tool_loop: bool,
}

/// Body of `POST /internal/tools`.
#[derive(Debug, Deserialize)]
pub struct ToolRoundRequest {
    context: HydrationContext,
    /// The upstream request body the caller just sent.
    request: Value,
    /// What the model returned for it.
    response: Value,
}

/// Body of `POST /internal/persist`.
#[derive(Debug, Deserialize)]
pub struct PersistRequest {
    context: HydrationContext,
    response: Value,
}

pub async fn internal_hydrate(
    State(state): State<AppState>,
    Query(query): Query<HydrateQuery>,
    req: Request,
) -> Response {
    let (_, body) = req.into_parts();
    let payload: RequestPayload = match read_json(body).await {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    match split::hydrate(payload, state.exec_ctx.as_ref(), query.tool_loop).await {
        Ok(hydration) => axum::Json(hydration).into_response(),
        Err(error) => executor_error_response(error),
    }
}

pub async fn internal_tools(State(state): State<AppState>, req: Request) -> Response {
    let (_, body) = req.into_parts();
    let request: ToolRoundRequest = match read_json(body).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    match split::tool_round(
        request.context,
        &request.request,
        &request.response,
        state.exec_ctx.as_ref(),
    )
    .await
    {
        Ok(round) => axum::Json(round).into_response(),
        Err(error) => executor_error_response(error),
    }
}

pub async fn internal_persist(State(state): State<AppState>, req: Request) -> Response {
    let (_, body) = req.into_parts();
    let request: PersistRequest = match read_json(body).await {
        Ok(request) => request,
        Err(response) => return response,
    };
    match split::persist(request.context, &request.response, state.exec_ctx.as_ref()).await {
        Ok(payload) => axum::Json(payload).into_response(),
        Err(error) => executor_error_response(error),
    }
}

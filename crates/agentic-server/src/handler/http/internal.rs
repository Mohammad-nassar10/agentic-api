//! Cluster-internal split-execution endpoints for an orchestrator that runs
//! inference itself. They trust their caller: restrict them at the network
//! layer, never expose them on a public route.

use axum::extract::{Request, State};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde_json::Value;

use agentic_core::executor::split;
use agentic_core::executor::split::HydrationContext;
use agentic_core::types::request_response::RequestPayload;

use super::super::common::{executor_error_response, read_json};
use crate::app::AppState;

/// Body of `POST /internal/persist`.
#[derive(Debug, Deserialize)]
pub struct PersistRequest {
    context: HydrationContext,
    response: Value,
}

pub async fn internal_hydrate(State(state): State<AppState>, req: Request) -> Response {
    let (_, body) = req.into_parts();
    let payload: RequestPayload = match read_json(body).await {
        Ok(payload) => payload,
        Err(response) => return response,
    };
    match split::hydrate(payload, state.exec_ctx.as_ref()).await {
        Ok(hydration) => axum::Json(hydration).into_response(),
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

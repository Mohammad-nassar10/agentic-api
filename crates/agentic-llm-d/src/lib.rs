//! Backend mode — agentic-api as state services for an orchestrator that runs
//! inference itself (the llm-d coordinator). Nothing here proxies or calls a model.

pub mod context;
pub mod handler;
pub mod runner;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};

use agentic_core::executor::ExecutionContext;

/// All the endpoints need. The gateway's `AppState` carries eight more fields,
/// every one for machinery backend mode does not run.
#[derive(Clone)]
pub struct InternalState {
    pub exec_ctx: Arc<ExecutionContext>,
    /// Seals the context between hydrate and persist, so a caller cannot forge
    /// one. Required: the endpoints have no other proof of where a turn began.
    pub signing_key: Arc<Vec<u8>>,
}

/// The whole surface: two split-execution endpoints and two probes.
pub fn build_internal_router(state: InternalState) -> Router {
    Router::new()
        .route("/health", get(handler::health))
        .route("/ready", get(handler::ready))
        .route("/internal/hydrate", post(handler::internal_hydrate))
        .route("/internal/persist", post(handler::internal_persist))
        .with_state(state)
}

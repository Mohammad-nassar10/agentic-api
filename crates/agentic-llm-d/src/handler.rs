//! The endpoints, and the axum glue they need. They trust their caller:
//! `hydrate` returns full conversation history and `persist` writes a turn from
//! a caller-supplied context, so bind this binary cluster-internal only.

use std::time::Duration;

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::value::RawValue;
use tracing::warn;

use agentic_core::executor::ExecutorError;
use agentic_core::executor::request::SplitContext;
use agentic_core::executor::split::{self, UpstreamBody};
use agentic_core::types::request_response::RequestPayload;

use crate::InternalState;

const MAX_BODY_SIZE: usize = 10 * 1024 * 1024;
/// Readiness means storage answers — llm-d owns the model fleet.
const STORAGE_PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Body of `POST /internal/persist`: the context, plus exactly one response form.
#[derive(Debug, Deserialize)]
pub struct PersistRequest {
    context: SplitContext,
    response: Option<Box<RawValue>>,
    sse: Option<String>,
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
    match split::hydrate(payload, state.exec_ctx.as_ref()).await {
        Ok(hydration) => axum::Json(hydration).into_response(),
        Err(error) => error_response(error),
    }
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
    match split::persist(context, upstream, state.exec_ctx.as_ref()).await {
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

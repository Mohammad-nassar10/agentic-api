//! Build state, bind, serve, drain.

use std::sync::Arc;

use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use tracing::info;

use agentic_core::config::Config;
use agentic_core::executor::ExecutionContext;

use crate::{InternalState, build_internal_router};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to build the execution context: {0}")]
    Context(#[from] agentic_core::error::Error),
    #[error("failed to bind or serve: {0}")]
    Io(#[from] std::io::Error),
}

/// Opens storage from `config` and serves the internal router until `shutdown`.
///
/// # Errors
/// If the storage pool cannot be opened, or the address cannot be bound.
pub async fn serve(
    config: &Config,
    signing_key: Vec<u8>,
    host: &str,
    port: u16,
    shutdown: CancellationToken,
) -> Result<(), Error> {
    let exec_ctx = Arc::new(ExecutionContext::from_config(config).await?);
    let signing_key = Arc::new(signing_key);
    let listener = TcpListener::bind(format!("{host}:{port}")).await?;
    info!("agentic-llm-d listening on {host}:{port} — no proxy, no inference");
    axum::serve(listener, build_internal_router(InternalState { exec_ctx, signing_key }))
        .with_graceful_shutdown(async move { shutdown.cancelled().await })
        .await?;
    Ok(())
}

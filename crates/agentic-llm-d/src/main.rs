//! `agentic-llm-d` — the coordinator calls `hydrate`, runs inference against its
//! own model fleet, then calls `persist`. This binary does neither.

use clap::Parser;
use tokio_util::sync::CancellationToken;

use agentic_core::config::{Config, PostgresConfig, SqliteConfig, ToolRuntimeConfig};
use agentic_llm_d::runner;

#[derive(Parser)]
#[command(name = "agentic-llm-d", about = "agentic-api backend mode for the llm-d coordinator")]
struct Cli {
    /// Keep this cluster-internal: the endpoints trust their caller.
    #[arg(long, env = "AGENTIC_LLM_D_HOST", default_value = "127.0.0.1")]
    host: String,
    #[arg(long, env = "AGENTIC_LLM_D_PORT", default_value_t = 8081)]
    port: u16,
    /// Defaults to the local database under the agentic-api home.
    #[arg(long, env = "DATABASE_URL")]
    db_url: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), runner::Error> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    // Every model-facing field is inert here; llm-d owns the fleet.
    let config = Config {
        llm_api_base: String::new(),
        openai_api_key: None,
        llm_ready_timeout_s: 0.0,
        llm_ready_interval_s: 0.0,
        skip_llm_ready_check: true,
        db_url: cli.db_url,
        postgres: PostgresConfig::default(),
        sqlite: SqliteConfig::default(),
        tools: ToolRuntimeConfig::default(),
    };

    let shutdown = CancellationToken::new();
    let on_signal = shutdown.clone();
    tokio::spawn(async move {
        if tokio::signal::ctrl_c().await.is_ok() {
            on_signal.cancel();
        }
    });

    runner::serve(&config, &cli.host, cli.port, shutdown).await
}

//! Optional context compaction through an external compaction service.
//!
//! The service takes and returns an OpenAI-shaped body: `{"model", "messages"}`
//! in, the same shape out with the messages folded. Deterministic modes rewrite
//! bulky tool output in place; LLM-backed modes summarise.
//!
//! Every failure path is fail-open — callers fall back to storing the history
//! unchanged, so a compactor outage costs compression rather than correctness.

use std::time::Duration;

use serde_json::{Value, json};
use tracing::{debug, warn};

/// Environment variable holding the compaction endpoint, e.g.
/// `http://context-guru:4000/compact`. Unset disables compaction.
pub const ADDRESS_ENV: &str = "COMPACTION_ADDRESS";

/// Compaction runs off the request path, but a wedged service should not keep
/// a background task alive indefinitely.
const TIMEOUT: Duration = Duration::from_secs(15);

/// Read the configured endpoint from the environment.
#[must_use]
pub fn address_from_env() -> Option<String> {
    std::env::var(ADDRESS_ENV)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

/// Fold a message history through the compaction service.
///
/// Returns `None` when the service is unreachable, answers with a non-success
/// status, or returns a body without a usable `messages` array. Callers treat
/// that as "store the history unchanged".
pub async fn compact(address: &str, client: &reqwest::Client, model: &str, messages: &[Value]) -> Option<Vec<Value>> {
    // Encoded by hand rather than with `RequestBuilder::json`: reqwest's `json`
    // feature is only a dev-dependency here, so using it would compile under
    // `--all-targets` and fail the real `cargo build -p agentic-server`.
    let payload = serde_json::to_vec(&json!({ "model": model, "messages": messages }))
        .map_err(|error| warn!("could not encode compaction request: {error}"))
        .ok()?;

    let response = client
        .post(address)
        .header("content-type", "application/json")
        .body(payload)
        .timeout(TIMEOUT)
        .send()
        .await
        .map_err(|error| warn!("compaction request to {address} failed: {error}"))
        .ok()?;

    let status = response.status();
    if !status.is_success() {
        warn!("compaction service returned {status}");
        return None;
    }

    let raw = response
        .bytes()
        .await
        .map_err(|error| warn!("could not read compaction response: {error}"))
        .ok()?;

    let body = serde_json::from_slice::<Value>(&raw)
        .map_err(|error| warn!("compaction response was not JSON: {error}"))
        .ok()?;

    let compacted = body.get("messages").and_then(Value::as_array).cloned();
    match compacted {
        // An empty result would wipe the prefix; treat it as unusable.
        Some(messages) if !messages.is_empty() => {
            debug!("compaction returned {} messages", messages.len());
            Some(messages)
        }
        _ => {
            warn!("compaction response had no usable messages array");
            None
        }
    }
}

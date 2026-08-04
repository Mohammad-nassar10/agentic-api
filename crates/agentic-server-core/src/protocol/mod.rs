//! Upstream inference protocol selection.
//!
//! The gateway's own client-facing API is always the Responses API. This module
//! selects the wire protocol used to talk to the inference backend *behind* it:
//! either the Responses API (vLLM's `/v1/responses`) or Chat Completions
//! (`/v1/chat/completions`), for backends and proxies that serve only the
//! latter.

use std::fmt;
use std::str::FromStr;

pub mod chat;

/// Wire protocol spoken to the upstream inference backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum UpstreamApi {
    /// `OpenAI` Responses API — the native protocol, requires no translation.
    #[default]
    Responses,
    /// `OpenAI` Chat Completions API. Requests and replies are translated by
    /// [`chat`]; streaming is not supported.
    ChatCompletions,
}

impl UpstreamApi {
    /// Path appended to the configured base URL for inference calls.
    #[must_use]
    pub fn inference_path(self) -> &'static str {
        match self {
            Self::Responses => "/v1/responses",
            Self::ChatCompletions => "/v1/chat/completions",
        }
    }

    #[must_use]
    pub fn is_chat_completions(self) -> bool {
        matches!(self, Self::ChatCompletions)
    }
}

impl fmt::Display for UpstreamApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Responses => "responses",
            Self::ChatCompletions => "chat_completions",
        };
        f.write_str(name)
    }
}

impl FromStr for UpstreamApi {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "responses" => Ok(Self::Responses),
            "chat_completions" | "chat" => Ok(Self::ChatCompletions),
            other => Err(format!(
                "unknown upstream API '{other}' (expected 'responses' or 'chat_completions')"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_known_names() {
        assert_eq!("responses".parse::<UpstreamApi>().unwrap(), UpstreamApi::Responses);
        assert_eq!(
            "chat_completions".parse::<UpstreamApi>().unwrap(),
            UpstreamApi::ChatCompletions
        );
        assert_eq!(
            "Chat-Completions".parse::<UpstreamApi>().unwrap(),
            UpstreamApi::ChatCompletions
        );
    }

    #[test]
    fn rejects_unknown_names() {
        assert!("completions".parse::<UpstreamApi>().is_err());
    }

    #[test]
    fn paths_match_protocol() {
        assert_eq!(UpstreamApi::Responses.inference_path(), "/v1/responses");
        assert_eq!(UpstreamApi::ChatCompletions.inference_path(), "/v1/chat/completions");
    }

    #[test]
    fn default_is_responses() {
        assert_eq!(UpstreamApi::default(), UpstreamApi::Responses);
    }
}

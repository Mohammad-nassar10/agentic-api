//! Chat Completions upstream adapter.
//!
//! Translates the gateway's Responses-shaped requests into Chat Completions
//! requests, and translates replies back into the Responses shape, so the rest
//! of the executor keeps working exclusively in Responses terms.
//!
//! Streaming is deliberately unsupported: Chat Completions emits
//! `chat.completion.chunk` deltas, which do not carry the response/item
//! envelope the streaming accumulator expects. [`request_json`] rejects it
//! rather than producing a silently malformed stream.

use serde_json::{Map, Value, json};

use crate::executor::error::{ExecutorError, ExecutorResult};
use crate::types::event::MessageStatus;
use crate::types::io::{
    FunctionToolCall, InputContent, InputItem, InputMessageContent, OutputItem, OutputMessage, OutputTextContent,
    ResponsesInput, ToolChoice,
};
use crate::types::request_response::{RequestPayload, UpstreamTool};
use crate::utils::common::{deserialize_from_str, serialize_to_string, serialize_to_value, uuid7_str};

/// Build a Chat Completions request body from a Responses request.
///
/// # Errors
///
/// Returns [`ExecutorError::InvalidRequest`] when `stream` is true, and
/// propagates tool-normalization failures from
/// [`RequestPayload::to_upstream_request`].
pub fn request_json(request: &RequestPayload, stream: bool) -> ExecutorResult<String> {
    if stream {
        return Err(ExecutorError::InvalidRequest(
            "streaming is not supported when the upstream API is 'chat_completions'".to_owned(),
        ));
    }

    // Reuse the Responses normalization first: it flattens Codex namespaces,
    // normalizes tool declarations, and folds compaction checkpoints into the
    // input. Only the wire shape differs from here on.
    let upstream = request.to_upstream_request(false)?;

    let mut messages: Vec<Value> = Vec::new();
    if let Some(instructions) = upstream.instructions {
        messages.push(json!({ "role": "system", "content": instructions }));
    }
    match upstream.input.as_ref() {
        ResponsesInput::Text(text) => messages.push(json!({ "role": "user", "content": text })),
        ResponsesInput::Items(items) => {
            for item in items {
                push_message(&mut messages, item);
            }
        }
    }

    let mut body = Map::new();
    body.insert("model".to_owned(), Value::String(upstream.model.to_owned()));
    body.insert("messages".to_owned(), Value::Array(messages));
    body.insert("stream".to_owned(), Value::Bool(false));

    // Responses caps output with `max_output_tokens`; Chat Completions uses
    // `max_tokens`.
    if let Some(max_output_tokens) = upstream.max_output_tokens {
        body.insert("max_tokens".to_owned(), json!(max_output_tokens));
    }
    if let Some(temperature) = upstream.temperature {
        body.insert("temperature".to_owned(), json!(temperature));
    }
    if let Some(top_p) = upstream.top_p {
        body.insert("top_p".to_owned(), json!(top_p));
    }
    if let Some(parallel_tool_calls) = upstream.parallel_tool_calls {
        body.insert("parallel_tool_calls".to_owned(), json!(parallel_tool_calls));
    }
    // `tool_choice` is only valid alongside a non-empty `tools` array — Chat
    // Completions rejects the request outright otherwise ("When using
    // `tool_choice`, `tools` must be set"). `to_upstream_request` always
    // resolves a choice, so the Responses path relies on
    // `is_absent_or_default_tool_choice` to drop the default; mirror both rules
    // here: emit only with tools, and never for the default `Auto`.
    let tools = upstream
        .tools
        .as_deref()
        .map(chat_tools)
        .filter(|tools| !tools.is_empty());
    if let Some(tools) = tools {
        body.insert("tools".to_owned(), Value::Array(tools));
        if let Some(tool_choice) = upstream
            .tool_choice
            .as_ref()
            .filter(|choice| !matches!(choice, ToolChoice::Auto))
            .and_then(chat_tool_choice)
        {
            body.insert("tool_choice".to_owned(), tool_choice);
        }
    }

    serialize_to_string(&Value::Object(body)).map_err(ExecutorError::JsonError)
}

/// Translate a Chat Completions response body into the Responses shape consumed
/// by [`crate::executor::accumulator::ResponseAccumulator::from_json`].
///
/// # Errors
///
/// Returns [`ExecutorError::JsonError`] if the body is not valid JSON, and
/// [`ExecutorError::ParseError`] if it lacks `id` or `choices`.
pub fn response_to_responses_json(body: &str) -> ExecutorResult<String> {
    let chat: Value = deserialize_from_str(body).map_err(ExecutorError::JsonError)?;

    let id = chat
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| ExecutorError::ParseError("missing 'id' field in chat completion response".to_owned()))?
        .to_owned();

    let choice = chat
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| ExecutorError::ParseError("chat completion response has no choices".to_owned()))?;

    let message = choice.get("message").unwrap_or(&Value::Null);
    let mut output: Vec<OutputItem> = Vec::new();

    if let Some(text) = message
        .get("content")
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
    {
        let mut assistant = OutputMessage::new(uuid7_str("msg_"), MessageStatus::Completed);
        assistant.content.push(OutputTextContent::new(text));
        output.push(OutputItem::Message(assistant));
    }

    if let Some(tool_calls) = message.get("tool_calls").and_then(Value::as_array) {
        for call in tool_calls {
            let function = call.get("function").unwrap_or(&Value::Null);
            output.push(OutputItem::FunctionCall(FunctionToolCall {
                // The Responses item id is ours; `call_id` is the correlation
                // key the client echoes back in `function_call_output`.
                id: uuid7_str("fc_"),
                call_id: string_at(call, "id"),
                name: string_at(function, "name"),
                namespace: None,
                arguments: string_at(function, "arguments"),
                status: MessageStatus::Completed,
            }));
        }
    }

    let truncated = choice.get("finish_reason").and_then(Value::as_str) == Some("length");

    let mut response = Map::new();
    response.insert("id".to_owned(), Value::String(id));
    response.insert(
        "output".to_owned(),
        serialize_to_value(&output).map_err(ExecutorError::JsonError)?,
    );
    response.insert(
        "status".to_owned(),
        Value::String(if truncated { "incomplete" } else { "completed" }.to_owned()),
    );
    if truncated {
        response.insert(
            "incomplete_details".to_owned(),
            json!({ "reason": "max_output_tokens" }),
        );
    }
    if let Some(usage) = chat_usage(chat.get("usage").unwrap_or(&Value::Null)) {
        response.insert("usage".to_owned(), usage);
    }

    serialize_to_string(&Value::Object(response)).map_err(ExecutorError::JsonError)
}

fn string_at(value: &Value, key: &str) -> String {
    value.get(key).and_then(Value::as_str).unwrap_or_default().to_owned()
}

fn push_message(messages: &mut Vec<Value>, item: &InputItem) {
    match item {
        InputItem::Message(message) => messages.push(json!({
            "role": message.role,
            "content": message_text(&message.content),
        })),
        InputItem::FunctionCall(call) => messages.push(json!({
            "role": "assistant",
            "content": Value::Null,
            "tool_calls": [{
                "id": call.call_id,
                "type": "function",
                "function": { "name": call.name, "arguments": call.arguments },
            }],
        })),
        InputItem::FunctionCallOutput(result) => messages.push(json!({
            "role": "tool",
            "tool_call_id": result.call_id,
            "content": result.output,
        })),
        // Reasoning items, freeform custom tool traffic, and compaction
        // checkpoints (already folded into messages by `model_input`) have no
        // Chat Completions representation.
        InputItem::CustomToolCall(_)
        | InputItem::CustomToolCallOutput(_)
        | InputItem::Reasoning(_)
        | InputItem::Compaction(_)
        | InputItem::Unknown => {}
    }
}

fn message_text(content: &InputMessageContent) -> String {
    match content {
        InputMessageContent::Text(text) => text.clone(),
        InputMessageContent::Parts(parts) => parts.iter().filter_map(part_text).collect::<Vec<_>>().join(""),
    }
}

fn part_text(part: &InputContent) -> Option<&str> {
    match part {
        InputContent::InputText(text) | InputContent::OutputText(text) | InputContent::ReasoningText(text) => {
            Some(text.text.as_str())
        }
        InputContent::InputImage(_) | InputContent::Unknown => None,
    }
}

fn chat_tools(tools: &[UpstreamTool]) -> Vec<Value> {
    tools
        .iter()
        .filter_map(|tool| match tool {
            UpstreamTool::Function(function) => Some(json!({
                "type": "function",
                "function": {
                    "name": function.name,
                    "description": function.description,
                    "parameters": function.parameters.clone().unwrap_or_else(|| json!({
                        "type": "object",
                        "properties": {},
                    })),
                },
            })),
            // Freeform custom tools are not expressible as Chat Completions
            // functions; dropping them is better than sending an invalid shape.
            UpstreamTool::Custom(_) => None,
        })
        .collect()
}

fn chat_tool_choice(choice: &ToolChoice) -> Option<Value> {
    match choice {
        ToolChoice::Auto => Some(Value::String("auto".to_owned())),
        ToolChoice::None => Some(Value::String("none".to_owned())),
        ToolChoice::Required => Some(Value::String("required".to_owned())),
        ToolChoice::Function { name, .. } => Some(json!({
            "type": "function",
            "function": { "name": name.as_str() },
        })),
        // No Chat Completions equivalent — fall back to the default behaviour.
        ToolChoice::Custom { .. } => None,
    }
}

fn chat_usage(usage: &Value) -> Option<Value> {
    if !usage.is_object() {
        return None;
    }
    let input_tokens = usage.get("prompt_tokens").and_then(Value::as_i64).unwrap_or_default();
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_i64)
        .unwrap_or(input_tokens + output_tokens);
    Some(json!({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(value: Value) -> RequestPayload {
        serde_json::from_value(value).expect("valid request payload")
    }

    fn body(request: &RequestPayload) -> Value {
        serde_json::from_str(&request_json(request, false).expect("request maps")).expect("valid json")
    }

    #[test]
    fn text_input_becomes_user_message() {
        let mapped = body(&request(json!({ "model": "m", "input": "hello" })));

        assert_eq!(mapped["messages"], json!([{ "role": "user", "content": "hello" }]));
        assert_eq!(mapped["model"], "m");
        assert_eq!(mapped["stream"], false);
    }

    #[test]
    fn instructions_become_leading_system_message() {
        let mapped = body(&request(json!({
            "model": "m",
            "instructions": "be terse",
            "input": "hello",
        })));

        assert_eq!(mapped["messages"][0], json!({ "role": "system", "content": "be terse" }));
        assert_eq!(mapped["messages"][1]["role"], "user");
    }

    #[test]
    fn max_output_tokens_becomes_max_tokens() {
        let mapped = body(&request(json!({
            "model": "m",
            "input": "hello",
            "max_output_tokens": 64,
        })));

        assert_eq!(mapped["max_tokens"], 64);
        assert!(mapped.get("max_output_tokens").is_none());
    }

    #[test]
    fn tool_call_history_becomes_assistant_and_tool_messages() {
        let mapped = body(&request(json!({
            "model": "m",
            "input": [
                { "type": "message", "role": "user", "content": "weather?" },
                { "type": "function_call", "call_id": "call_1", "name": "get_weather", "arguments": "{}" },
                { "type": "function_call_output", "call_id": "call_1", "output": "sunny" },
            ],
        })));

        let messages = mapped["messages"].as_array().expect("messages array");
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[1]["role"], "assistant");
        assert_eq!(messages[1]["tool_calls"][0]["id"], "call_1");
        assert_eq!(messages[1]["tool_calls"][0]["function"]["name"], "get_weather");
        assert_eq!(
            messages[2],
            json!({ "role": "tool", "tool_call_id": "call_1", "content": "sunny" })
        );
    }

    #[test]
    fn function_tools_are_nested_under_function_key() {
        let mapped = body(&request(json!({
            "model": "m",
            "input": "hello",
            "tools": [{
                "type": "function",
                "name": "get_weather",
                "description": "look up weather",
                "parameters": { "type": "object", "properties": {} },
            }],
        })));

        assert_eq!(mapped["tools"][0]["type"], "function");
        assert_eq!(mapped["tools"][0]["function"]["name"], "get_weather");
        assert_eq!(mapped["tools"][0]["function"]["description"], "look up weather");
    }

    /// vLLM rejects `tool_choice` without `tools` ("When using `tool_choice`,
    /// `tools` must be set"), and `to_upstream_request` always resolves a
    /// choice — so a toolless request must not carry one.
    #[test]
    fn toolless_request_omits_tool_choice() {
        let mapped = body(&request(json!({ "model": "m", "input": "hello" })));

        assert!(mapped.get("tool_choice").is_none(), "got {mapped}");
        assert!(mapped.get("tools").is_none(), "got {mapped}");
    }

    #[test]
    fn default_auto_tool_choice_is_omitted_even_with_tools() {
        let mapped = body(&request(json!({
            "model": "m",
            "input": "hello",
            "tool_choice": "auto",
            "tools": [{ "type": "function", "name": "f", "parameters": {} }],
        })));

        assert!(mapped.get("tools").is_some());
        assert!(mapped.get("tool_choice").is_none(), "got {mapped}");
    }

    #[test]
    fn explicit_tool_choice_is_forwarded_with_tools() {
        let mapped = body(&request(json!({
            "model": "m",
            "input": "hello",
            "tool_choice": "required",
            "tools": [{ "type": "function", "name": "f", "parameters": {} }],
        })));

        assert_eq!(mapped["tool_choice"], "required");
    }

    #[test]
    fn streaming_is_rejected() {
        let error = request_json(&request(json!({ "model": "m", "input": "hi" })), true).unwrap_err();

        assert!(error.to_string().contains("streaming is not supported"));
    }

    #[test]
    fn response_content_becomes_output_message() {
        let mapped: Value = serde_json::from_str(
            &response_to_responses_json(
                &json!({
                    "id": "chatcmpl-1",
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": "hi there" },
                        "finish_reason": "stop",
                    }],
                    "usage": { "prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16 },
                })
                .to_string(),
            )
            .expect("response maps"),
        )
        .expect("valid json");

        assert_eq!(mapped["id"], "chatcmpl-1");
        assert_eq!(mapped["status"], "completed");
        assert_eq!(mapped["output"][0]["type"], "message");
        assert_eq!(mapped["output"][0]["content"][0]["text"], "hi there");
        assert_eq!(mapped["usage"], json!({
            "input_tokens": 10,
            "output_tokens": 6,
            "total_tokens": 16,
        }));
    }

    #[test]
    fn length_finish_reason_marks_response_incomplete() {
        let mapped: Value = serde_json::from_str(
            &response_to_responses_json(
                &json!({
                    "id": "chatcmpl-1",
                    "choices": [{ "message": { "content": "tru" }, "finish_reason": "length" }],
                })
                .to_string(),
            )
            .expect("response maps"),
        )
        .expect("valid json");

        assert_eq!(mapped["status"], "incomplete");
        assert_eq!(mapped["incomplete_details"]["reason"], "max_output_tokens");
    }

    #[test]
    fn response_tool_calls_become_function_call_items() {
        let mapped: Value = serde_json::from_str(
            &response_to_responses_json(
                &json!({
                    "id": "chatcmpl-1",
                    "choices": [{
                        "message": {
                            "content": Value::Null,
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": { "name": "get_weather", "arguments": "{\"city\":\"NYC\"}" },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                })
                .to_string(),
            )
            .expect("response maps"),
        )
        .expect("valid json");

        assert_eq!(mapped["output"][0]["type"], "function_call");
        assert_eq!(mapped["output"][0]["call_id"], "call_1");
        assert_eq!(mapped["output"][0]["name"], "get_weather");
    }

    /// The accumulator drops output items it cannot deserialize, so a shape
    /// mistake here would surface as a silently empty response rather than an
    /// error. Round-trip through `OutputItem` to catch that.
    #[test]
    fn mapped_output_items_survive_deserialization() {
        let mapped: Value = serde_json::from_str(
            &response_to_responses_json(
                &json!({
                    "id": "chatcmpl-1",
                    "choices": [{
                        "message": {
                            "content": "hi",
                            "tool_calls": [{
                                "id": "call_1",
                                "function": { "name": "f", "arguments": "{}" },
                            }],
                        },
                        "finish_reason": "stop",
                    }],
                })
                .to_string(),
            )
            .expect("response maps"),
        )
        .expect("valid json");

        let items: Vec<OutputItem> = serde_json::from_value(mapped["output"].clone()).expect("output items parse");

        assert_eq!(items.len(), 2);
        assert!(matches!(items[0], OutputItem::Message(_)));
        assert!(matches!(items[1], OutputItem::FunctionCall(_)));
    }

    #[test]
    fn missing_choices_is_a_parse_error() {
        let error = response_to_responses_json(&json!({ "id": "chatcmpl-1" }).to_string()).unwrap_err();

        assert!(error.to_string().contains("no choices"));
    }
}

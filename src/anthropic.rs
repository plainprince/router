//! Translation between the OpenAI chat-completions dialect (what clients speak)
//! and the Anthropic messages API (what `anthropic` providers speak).
//!
//! Requests are translated to `/v1/messages`; responses are translated back to
//! `chat.completion` objects. Streaming responses are translated event by event
//! so text still arrives as it is generated.

use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use bytes::Bytes;
use serde_json::{json, Value};

pub const ANTHROPIC_VERSION: &str = "2023-06-01";
/// Anthropic requires `max_tokens`; this is used when the client omits it.
pub const DEFAULT_MAX_TOKENS: u64 = 8192;

/// Build an Anthropic `/v1/messages` body from an OpenAI chat request.
pub fn request_body(openai: &Value, model: &str) -> Result<Value, String> {
    let messages = openai
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` must be an array")?;

    let mut system: Vec<String> = Vec::new();
    let mut translated: Vec<Value> = Vec::new();
    let mut generated_tool_ids = 0usize;

    for message in messages {
        match message.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => {
                if let Some(text) = content_as_text(message.get("content")) {
                    system.push(text);
                }
            }
            "user" => {
                let blocks = content_blocks(message.get("content"))?;
                push_message(&mut translated, "user", blocks);
            }
            "assistant" => {
                let mut blocks = content_blocks(message.get("content"))?;
                if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let function = call
                            .get("function")
                            .ok_or("tool calls need a `function` object")?;
                        let name = function
                            .get("name")
                            .and_then(Value::as_str)
                            .ok_or("tool calls need `function.name`")?;
                        let id = match call.get("id").and_then(Value::as_str) {
                            Some(id) if !id.is_empty() => id.to_owned(),
                            _ => {
                                generated_tool_ids += 1;
                                format!("toolu_router_{generated_tool_ids}")
                            }
                        };
                        blocks.push(json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": tool_arguments(function.get("arguments")),
                        }));
                    }
                }
                push_message(&mut translated, "assistant", blocks);
            }
            "tool" => {
                let tool_use_id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .ok_or("`tool` messages need a `tool_call_id`")?;
                let text = content_as_text(message.get("content")).unwrap_or_default();
                push_message(
                    &mut translated,
                    "user",
                    vec![json!({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": text,
                    })],
                );
            }
            other => return Err(format!("unsupported message role `{other}`")),
        }
    }

    let mut body = json!({
        "model": model,
        "max_tokens": max_tokens(openai),
        "messages": translated,
    });
    if !system.is_empty() {
        body["system"] = Value::String(system.join("\n\n"));
    }
    if openai.get("stream").and_then(Value::as_bool).unwrap_or(false) {
        body["stream"] = Value::Bool(true);
    }
    for key in ["temperature", "top_p"] {
        if let Some(value) = openai.get(key).filter(|v| v.is_number()) {
            body[key] = value.clone();
        }
    }
    if let Some(stop) = stop_sequences(openai.get("stop")) {
        body["stop_sequences"] = Value::Array(stop);
    }
    if let Some((tools, choice)) = tools(openai)? {
        body["tools"] = Value::Array(tools);
        if let Some(choice) = choice {
            body["tool_choice"] = choice;
        }
    }
    Ok(body)
}

/// Translate an Anthropic `message` (or `error`) into an OpenAI completion.
pub fn response_body(message: &Value, model: &str) -> Value {
    if let Some(error) = message.get("error") {
        return error_body(error);
    }

    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in message
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match block.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(part) = block.get("text").and_then(Value::as_str) {
                    text.push_str(part);
                }
            }
            Some("tool_use") => {
                let index = tool_calls.len();
                tool_calls.push(json!({
                    "id": block.get("id").and_then(Value::as_str)
                        .map_or_else(|| format!("call_{index}"), str::to_owned),
                    "type": "function",
                    "function": {
                        "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                        "arguments": block.get("input").map(Value::to_string).unwrap_or_else(|| "{}".to_owned()),
                    },
                }));
            }
            _ => {}
        }
    }

    let mut response_message = json!({
        "role": "assistant",
        "content": if text.is_empty() && !tool_calls.is_empty() {
            Value::Null
        } else {
            Value::String(text)
        },
    });
    if !tool_calls.is_empty() {
        response_message["tool_calls"] = Value::Array(tool_calls);
    }

    let usage = message.get("usage");
    let prompt_tokens = usage
        .and_then(|u| u.get("input_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let completion_tokens = usage
        .and_then(|u| u.get("output_tokens"))
        .and_then(Value::as_u64)
        .unwrap_or(0);

    json!({
        "id": message.get("id").and_then(Value::as_str).unwrap_or("chatcmpl-router"),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{
            "index": 0,
            "message": response_message,
            "finish_reason": finish_reason(
                message.get("stop_reason").and_then(Value::as_str)
            ),
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })
}

/// Translate an Anthropic SSE stream into OpenAI `chat.completion.chunk` events.
pub fn stream(
    response: reqwest::Response,
    model: String,
    include_usage: bool,
) -> impl futures_core::Stream<Item = Result<Bytes, std::io::Error>> + Send + 'static {
    use futures_util::StreamExt;

    let mut upstream = response.bytes_stream();
    let mut decoder = SseDecoder::default();
    let mut state = StreamState::new(model, include_usage);

    async_stream::stream! {
        while let Some(chunk) = upstream.next().await {
            match chunk {
                Ok(bytes) => {
                    for data in decoder.push(&bytes) {
                        for event in state.handle(&data) {
                            yield Ok(event);
                        }
                    }
                }
                Err(err) => yield Ok(state.fail(&format!("provider stream failed: {err}"))),
            }
            if state.done {
                break;
            }
        }
        if !state.done {
            yield Ok(state.finish());
        }
    }
}

/// Incremental SSE translation state, exposed for tests.
#[derive(Debug)]
struct StreamState {
    model: String,
    id: String,
    created: u64,
    include_usage: bool,
    prompt_tokens: u64,
    /// Anthropic content-block index -> OpenAI tool-call index.
    tool_calls: HashMap<u64, u64>,
    next_tool_call: u64,
    role_sent: bool,
    done: bool,
}

impl StreamState {
    fn new(model: String, include_usage: bool) -> Self {
        Self {
            model,
            id: "chatcmpl-router".to_owned(),
            created: now_secs(),
            include_usage,
            prompt_tokens: 0,
            tool_calls: HashMap::new(),
            next_tool_call: 0,
            role_sent: false,
            done: false,
        }
    }

    fn handle(&mut self, data: &str) -> Vec<Bytes> {
        if data.is_empty() || data == "[DONE]" {
            return Vec::new();
        }
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            return Vec::new();
        };

        match event.get("type").and_then(Value::as_str) {
            Some("message_start") => {
                if let Some(message) = event.get("message") {
                    if let Some(id) = message.get("id").and_then(Value::as_str) {
                        self.id = id.to_owned();
                    }
                    self.prompt_tokens = message
                        .get("usage")
                        .and_then(|u| u.get("input_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                }
                if self.role_sent {
                    return Vec::new();
                }
                self.role_sent = true;
                vec![self.chunk(json!({"role": "assistant", "content": ""}), None, None)]
            }
            Some("content_block_start") => {
                let Some(block) = event.get("content_block") else {
                    return Vec::new();
                };
                if block.get("type").and_then(Value::as_str) != Some("tool_use") {
                    return Vec::new();
                }
                let block_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                let call_index = self.next_tool_call;
                self.next_tool_call += 1;
                self.tool_calls.insert(block_index, call_index);
                vec![self.chunk(
                    json!({"tool_calls": [{
                        "index": call_index,
                        "id": block.get("id").and_then(Value::as_str).unwrap_or_default(),
                        "type": "function",
                        "function": {
                            "name": block.get("name").and_then(Value::as_str).unwrap_or_default(),
                            "arguments": "",
                        },
                    }]}),
                    None,
                    None,
                )]
            }
            Some("content_block_delta") => {
                let Some(delta) = event.get("delta") else {
                    return Vec::new();
                };
                match delta.get("type").and_then(Value::as_str) {
                    Some("text_delta") => vec![self.chunk(
                        json!({"content": delta.get("text").and_then(Value::as_str).unwrap_or_default()}),
                        None,
                        None,
                    )],
                    Some("input_json_delta") => {
                        let block_index = event.get("index").and_then(Value::as_u64).unwrap_or(0);
                        let Some(&call_index) = self.tool_calls.get(&block_index) else {
                            return Vec::new();
                        };
                        vec![self.chunk(
                            json!({"tool_calls": [{
                                "index": call_index,
                                "function": {
                                    "arguments": delta.get("partial_json").and_then(Value::as_str).unwrap_or_default(),
                                },
                            }]}),
                            None,
                            None,
                        )]
                    }
                    _ => Vec::new(),
                }
            }
            Some("message_delta") => {
                let stop_reason = event
                    .get("delta")
                    .and_then(|d| d.get("stop_reason"))
                    .and_then(Value::as_str);
                let usage = self.include_usage.then(|| {
                    let completion = event
                        .get("usage")
                        .and_then(|u| u.get("output_tokens"))
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    json!({
                        "prompt_tokens": self.prompt_tokens,
                        "completion_tokens": completion,
                        "total_tokens": self.prompt_tokens + completion,
                    })
                });
                vec![self.chunk(json!({}), Some(finish_reason(stop_reason)), usage)]
            }
            Some("message_stop") => {
                self.done = true;
                vec![Bytes::from_static(b"data: [DONE]\n\n")]
            }
            Some("error") => {
                let error = event.get("error").cloned().unwrap_or(Value::Null);
                vec![self.fail(
                    error
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("provider error"),
                )]
            }
            _ => Vec::new(),
        }
    }

    fn fail(&mut self, message: &str) -> Bytes {
        self.done = true;
        Bytes::from(format!(
            "data: {}\n\n",
            error_body(&json!({"message": message}))
        ))
    }

    fn finish(&mut self) -> Bytes {
        self.done = true;
        Bytes::from_static(b"data: [DONE]\n\n")
    }

    fn chunk(&self, delta: Value, finish: Option<&str>, usage: Option<Value>) -> Bytes {
        let mut payload = json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        });
        if let Some(usage) = usage {
            payload["usage"] = usage;
        }
        Bytes::from(format!("data: {payload}\n\n"))
    }
}

/// Splits an SSE byte stream into complete event payloads.
#[derive(Default)]
struct SseDecoder {
    buffer: Vec<u8>,
}

impl SseDecoder {
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buffer.extend_from_slice(chunk);
        let mut payloads = Vec::new();
        while let Some((end, next)) = event_boundary(&self.buffer) {
            let block = String::from_utf8_lossy(&self.buffer[..end]).into_owned();
            self.buffer.drain(..next);
            if let Some(data) = data_of(&block) {
                payloads.push(data);
            }
        }
        payloads
    }
}

fn event_boundary(buffer: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i + 1 < buffer.len() {
        if buffer[i] == b'\n' && buffer[i + 1] == b'\n' {
            return Some((i, i + 2));
        }
        if buffer[i] == b'\r' && i + 3 < buffer.len() && &buffer[i..i + 4] == b"\r\n\r\n" {
            return Some((i, i + 4));
        }
        i += 1;
    }
    None
}

fn data_of(block: &str) -> Option<String> {
    let data: Vec<&str> = block
        .split('\n')
        .filter_map(|line| {
            let line = line.strip_suffix('\r').unwrap_or(line);
            let value = line.strip_prefix("data:")?;
            Some(value.strip_prefix(' ').unwrap_or(value))
        })
        .collect();
    (!data.is_empty()).then(|| data.join("\n"))
}

fn finish_reason(stop_reason: Option<&str>) -> &'static str {
    match stop_reason {
        Some("max_tokens") => "length",
        Some("tool_use") => "tool_calls",
        Some("refusal") => "content_filter",
        _ => "stop",
    }
}

fn error_body(error: &Value) -> Value {
    let kind = error
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("provider_error");
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .unwrap_or("upstream provider error");
    json!({"error": {"message": message, "type": kind, "code": kind}})
}

fn max_tokens(openai: &Value) -> u64 {
    ["max_completion_tokens", "max_tokens"]
        .iter()
        .find_map(|key| openai.get(*key).and_then(Value::as_u64))
        .unwrap_or(DEFAULT_MAX_TOKENS)
}

fn stop_sequences(stop: Option<&Value>) -> Option<Vec<Value>> {
    match stop? {
        Value::String(sequence) if !sequence.is_empty() => {
            Some(vec![Value::String(sequence.clone())])
        }
        value @ Value::Array(_) => {
            let sequences: Vec<Value> = value
                .as_array()?
                .iter()
                .filter_map(Value::as_str)
                .map(|sequence| Value::String(sequence.to_owned()))
                .collect();
            (!sequences.is_empty()).then_some(sequences)
        }
        _ => None,
    }
}

fn tools(openai: &Value) -> Result<Option<(Vec<Value>, Option<Value>)>, String> {
    let Some(tools) = openai.get("tools").and_then(Value::as_array) else {
        return Ok(None);
    };
    let choice = openai.get("tool_choice");
    // Anthropic has no "never call a tool"; omitting the tools is the equivalent.
    if tools.is_empty() || choice.and_then(Value::as_str) == Some("none") {
        return Ok(None);
    }

    let mut converted = Vec::with_capacity(tools.len());
    for tool in tools {
        let function = tool
            .get("function")
            .ok_or("`tools` entries must be `{type: \"function\", function: {...}}`")?;
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .ok_or("tools need `function.name`")?;
        let mut converted_tool = json!({
            "name": name,
            "input_schema": function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({"type": "object", "properties": {}})),
        });
        if let Some(description) = function.get("description").and_then(Value::as_str) {
            converted_tool["description"] = Value::String(description.to_owned());
        }
        converted.push(converted_tool);
    }

    let choice = match choice {
        None => None,
        Some(Value::String(choice)) => match choice.as_str() {
            "auto" => Some(json!({"type": "auto"})),
            "required" | "any" => Some(json!({"type": "any"})),
            other => return Err(format!("unsupported tool_choice `{other}`")),
        },
        Some(value) => match value.get("type").and_then(Value::as_str) {
            Some("function") => {
                let name = value
                    .get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .ok_or("`tool_choice.function.name` is required")?;
                Some(json!({"type": "tool", "name": name}))
            }
            Some(other) => return Err(format!("unsupported tool_choice type `{other}`")),
            None => return Err("`tool_choice` must be a string or an object".to_owned()),
        },
    };

    Ok(Some((converted, choice)))
}

fn content_blocks(content: Option<&Value>) -> Result<Vec<Value>, String> {
    match content {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::String(text)) => Ok(if text.is_empty() {
            Vec::new()
        } else {
            vec![json!({"type": "text", "text": text})]
        }),
        Some(Value::Array(parts)) => {
            let mut blocks = Vec::with_capacity(parts.len());
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => {
                        if let Some(text) = part.get("text").and_then(Value::as_str) {
                            blocks.push(json!({"type": "text", "text": text}));
                        }
                    }
                    Some("image_url") => blocks.push(image_block(part)?),
                    Some(other) => {
                        return Err(format!("content part `{other}` is not supported"));
                    }
                    None => {}
                }
            }
            Ok(blocks)
        }
        Some(_) => Err("message content must be a string or an array of parts".to_owned()),
    }
}

fn image_block(part: &Value) -> Result<Value, String> {
    let url = part
        .get("image_url")
        .and_then(|image| image.get("url"))
        .and_then(Value::as_str)
        .ok_or("image parts need `image_url.url`")?;
    let source = match url.strip_prefix("data:") {
        Some(rest) => {
            let (meta, data) = rest.split_once(',').ok_or("malformed data URL in `image_url`")?;
            let media_type = meta.strip_suffix(";base64").unwrap_or(meta);
            json!({"type": "base64", "media_type": media_type, "data": data})
        }
        None => json!({"type": "url", "url": url}),
    };
    Ok(json!({"type": "image", "source": source}))
}

fn content_as_text(content: Option<&Value>) -> Option<String> {
    match content? {
        Value::String(text) => Some(text.clone()),
        Value::Array(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join("\n");
            (!text.is_empty()).then_some(text)
        }
        _ => None,
    }
}

fn tool_arguments(arguments: Option<&Value>) -> Value {
    match arguments {
        Some(Value::String(raw)) => serde_json::from_str(raw)
            .ok()
            .filter(Value::is_object)
            .unwrap_or_else(|| json!({})),
        Some(value) if value.is_object() => value.clone(),
        _ => json!({}),
    }
}

/// Append blocks to the previous message when the role matches (Anthropic wants
/// alternating turns, and tool results belong with the user message).
fn push_message(messages: &mut Vec<Value>, role: &str, blocks: Vec<Value>) {
    if blocks.is_empty() {
        return;
    }
    if let Some(last) = messages.last_mut() {
        if last.get("role").and_then(Value::as_str) == Some(role) {
            if let Some(content) = last.get_mut("content").and_then(Value::as_array_mut) {
                content.extend(blocks);
                return;
            }
        }
    }
    messages.push(json!({"role": role, "content": blocks}));
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translates_system_messages_and_sampling() {
        let request = json!({
            "model": "anthropic/claude-sonnet-4-5",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
            "temperature": 0.3,
            "stop": ["END"],
        });

        let body = request_body(&request, "claude-sonnet-4-5").unwrap();
        assert_eq!(body["model"], "claude-sonnet-4-5");
        assert_eq!(body["system"], "be terse");
        assert_eq!(body["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(body["temperature"], 0.3);
        assert_eq!(body["stop_sequences"][0], "END");
        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["content"][0]["text"], "hi");
    }

    #[test]
    fn translates_tool_calls_and_tool_results() {
        let request = json!({
            "messages": [
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": null, "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"},
                }]},
                {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            ],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Look up the weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                },
            }],
            "tool_choice": "required",
        });

        let body = request_body(&request, "claude-sonnet-4-5").unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages[1]["content"][0]["type"], "tool_use");
        assert_eq!(messages[1]["content"][0]["input"]["city"], "Paris");
        assert_eq!(messages[2]["content"][0]["type"], "tool_result");
        assert_eq!(messages[2]["content"][0]["tool_use_id"], "call_1");
        assert_eq!(body["tools"][0]["name"], "get_weather");
        assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
        assert_eq!(body["tool_choice"]["type"], "any");
    }

    #[test]
    fn merges_adjacent_messages_of_the_same_role() {
        let request = json!({
            "messages": [
                {"role": "user", "content": "one"},
                {"role": "user", "content": "two"},
            ],
        });
        let body = request_body(&request, "claude-sonnet-4-5").unwrap();
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["content"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn rejects_requests_without_messages() {
        let err = request_body(&json!({"model": "x"}), "x").unwrap_err();
        assert!(err.contains("messages"));
    }

    #[test]
    fn translates_a_message_response() {
        let message = json!({
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": [
                {"type": "text", "text": "Sure."},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 11, "output_tokens": 7},
        });

        let response = response_body(&message, "claude-sonnet-4-5");
        assert_eq!(response["id"], "msg_123");
        assert_eq!(response["object"], "chat.completion");
        assert_eq!(response["choices"][0]["message"]["content"], "Sure.");
        assert_eq!(response["choices"][0]["finish_reason"], "tool_calls");
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
            "search"
        );
        assert_eq!(
            response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"],
            "{\"q\":\"x\"}"
        );
        assert_eq!(response["usage"]["total_tokens"], 18);
    }

    #[test]
    fn translates_anthropic_errors() {
        let response = response_body(
            &json!({"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}),
            "claude-sonnet-4-5",
        );
        assert_eq!(response["error"]["message"], "bad");
        assert_eq!(response["error"]["type"], "invalid_request_error");
    }

    #[test]
    fn splits_sse_events_across_chunks() {
        let mut decoder = SseDecoder::default();
        assert!(decoder.push(b"event: content_block_delta\ndata: {\"a\":").is_empty());
        let payloads = decoder.push(b"1}\n\ndata: {\"b\":2}\n\n");
        assert_eq!(payloads, vec!["{\"a\":1}".to_owned(), "{\"b\":2}".to_owned()]);
        assert!(decoder.push(b"data: {\"partial\":").is_empty());
    }

    #[test]
    fn translates_streamed_text_and_tool_calls() {
        let mut state = StreamState::new("claude-sonnet-4-5".to_owned(), true);

        let start = state.handle(
            r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":5}}}"#,
        );
        assert_eq!(start.len(), 1);
        assert!(String::from_utf8_lossy(&start[0]).contains("\"role\":\"assistant\""));

        let text = state.handle(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
        );
        assert!(String::from_utf8_lossy(&text[0]).contains("\"content\":\"Hel\""));

        let tool = state.handle(
            r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"search"}}"#,
        );
        assert!(String::from_utf8_lossy(&tool[0]).contains("\"index\":0"));
        assert!(String::from_utf8_lossy(&tool[0]).contains("\"name\":\"search\""));

        // The content-block index (1) maps to tool-call index 0.
        let arguments = state.handle(
            r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"q\":"}}"#,
        );
        assert!(
            String::from_utf8_lossy(&arguments[0]).contains("\"index\":0"),
            "{}",
            String::from_utf8_lossy(&arguments[0])
        );

        let end = state.handle(
            r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}"#,
        );
        let end = String::from_utf8_lossy(&end[0]).to_owned();
        assert!(end.contains("\"finish_reason\":\"tool_calls\""), "{end}");
        assert!(end.contains("\"completion_tokens\":9"), "{end}");

        let stop = state.handle(r#"{"type":"message_stop"}"#);
        assert_eq!(stop[0], Bytes::from_static(b"data: [DONE]\n\n"));
        assert!(state.done);
    }

    #[test]
    fn ignores_ping_and_reports_stream_errors() {
        let mut state = StreamState::new("m".to_owned(), false);
        assert!(state.handle(r#"{"type":"ping"}"#).is_empty());
        let error = state.handle(r#"{"type":"error","error":{"message":"overloaded"}}"#);
        assert!(String::from_utf8_lossy(&error[0]).contains("overloaded"));
        assert!(state.done);
    }
}

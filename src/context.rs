//! Keeps a request inside a caller-supplied context window.
//!
//! A client may send `"max_context": <tokens>` in the request body. Before any
//! provider attempt the router removes that field, drops the oldest conversation
//! turns until the prompt fits, trims a single oversized message when it still
//! does not, and clamps `max_tokens` so input plus output stay under the limit.
//!
//! Tokens are estimated at four characters per token, which is close enough on
//! English prose and errs on the safe side for code (more tokens per character).
//! The goal is to keep a request from being rejected by a free-tier per-request
//! cap, not to do exact token accounting.

use serde_json::{Value, json};

/// Request field that carries the limit. Removed before anything is forwarded.
pub const FIELD: &str = "max_context";

/// Output room kept aside when the client did not ask for a specific budget.
const DEFAULT_OUTPUT_TOKENS: u64 = 1024;
/// Never squeeze the prompt below this, even for an absurdly small limit.
const MIN_INPUT_TOKENS: u64 = 256;
const TRUNCATION_NOTE: &str = "[…truncated to fit the context limit]";

/// What [`apply`] changed, for logs and response headers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Trim {
    /// Messages dropped from the front of the conversation.
    pub dropped_messages: usize,
    /// Estimated tokens removed by dropping and trimming.
    pub dropped_tokens: u64,
    /// Estimated prompt tokens before trimming.
    pub before_tokens: u64,
    /// Estimated prompt tokens after trimming.
    pub after_tokens: u64,
}

impl Trim {
    pub fn changed(self) -> bool {
        self.dropped_tokens > 0
    }
}

/// Enforce `max_context` on an OpenAI-shaped request body. Returns `None` when
/// the request does not ask for a limit.
pub fn apply(body: &mut Value) -> Option<Trim> {
    let object = body.as_object_mut()?;
    let limit = match object.remove(FIELD)? {
        Value::Number(number) => number.as_u64(),
        Value::String(text) => text.trim().parse::<u64>().ok(),
        _ => None,
    }
    .filter(|limit| *limit > 0)?;

    let tools_tokens = object
        .get("tools")
        .map(|tools| estimate(&tools.to_string()))
        .unwrap_or(0);
    let reserve = object
        .get("max_tokens")
        .and_then(Value::as_u64)
        .unwrap_or(DEFAULT_OUTPUT_TOKENS)
        .min(limit / 2)
        .max(1);
    // Tool definitions are part of the fixed prompt cost, so the conversation
    // gets whatever the window leaves after output room and the tools.
    // The router prepends its own runtime block (date, model) after this, so its
    // room comes out of the same budget.
    let budget = limit
        .saturating_sub(reserve)
        .saturating_sub(tools_tokens)
        .saturating_sub(crate::runtime::RESERVE_TOKENS)
        .max(MIN_INPUT_TOKENS.min(limit));

    let messages = object.get_mut("messages")?.as_array_mut()?;
    if messages.is_empty() {
        return None;
    }
    let mut total = messages.iter().map(message_tokens).sum::<u64>();
    let before_tokens = tools_tokens + total;
    let mut dropped_messages = 0;

    // Oldest non-system turn first; the system prompt and the newest message
    // always survive, because together they carry the actual request.
    while total > budget {
        let Some(index) = messages
            .iter()
            .position(|message| !is_system(message))
            .filter(|index| *index + 1 < messages.len())
        else {
            break;
        };
        // A tool call and its results are one turn: dropping only half of it
        // leaves a conversation providers reject.
        let mut count = 1;
        if messages[index].get("tool_calls").is_some() {
            while index + count < messages.len() - 1
                && messages[index + count].get("role").and_then(Value::as_str) == Some("tool")
            {
                count += 1;
            }
        }
        count = count.min(messages.len() - 1 - index);
        if count == 0 {
            break;
        }
        let removed: u64 = messages[index..index + count]
            .iter()
            .map(message_tokens)
            .sum();
        messages.drain(index..index + count);
        total = total.saturating_sub(removed);
        dropped_messages += count;
    }

    // One remaining old turn can still be huge (a pasted document, a long
    // answer). Shrink the largest non-final message until the prompt fits.
    while total > budget && messages.len() > 1 {
        let Some((index, size)) = messages
            .iter()
            .enumerate()
            .take(messages.len() - 1)
            .map(|(index, message)| (index, message_tokens(message)))
            .max_by_key(|(_, size)| *size)
        else {
            break;
        };
        if size <= 8 {
            break;
        }
        let target = size.saturating_sub(total - budget).max(8);
        if !trim_message(&mut messages[index], target) {
            break;
        }
        let now = message_tokens(&messages[index]);
        if now >= size {
            break;
        }
        total = total - size + now;
    }

    // The final message itself (often one long user prompt) may be over budget.
    if total > budget && !messages.is_empty() {
        let last = messages.len() - 1;
        let size = message_tokens(&messages[last]);
        if size > budget {
            let target = budget.max(8);
            if trim_message(&mut messages[last], target) {
                total = total - size + message_tokens(&messages[last]);
            }
        }
    }

    // Do not ask for more output than the input leaves room for.
    if let Some(max_tokens) = object.get("max_tokens").and_then(Value::as_u64) {
        let room = limit.saturating_sub(tools_tokens + total).max(1);
        if max_tokens > room {
            object.insert("max_tokens".to_owned(), json!(room));
        }
    }

    Some(Trim {
        dropped_messages,
        dropped_tokens: before_tokens.saturating_sub(tools_tokens + total),
        before_tokens,
        after_tokens: tools_tokens + total,
    })
}

fn is_system(message: &Value) -> bool {
    matches!(
        message.get("role").and_then(Value::as_str),
        Some("system") | Some("developer")
    )
}

/// Rough token count: four characters per token, plus per-message overhead.
pub fn estimate(text: &str) -> u64 {
    ((text.chars().count() as u64) + 3) / 4
}

fn message_tokens(message: &Value) -> u64 {
    let mut tokens = 4;
    match message.get("content") {
        Some(Value::String(text)) => tokens += estimate(text),
        Some(Value::Array(parts)) => {
            for part in parts {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    tokens += estimate(text);
                } else {
                    // Images and other non-text parts still cost something.
                    tokens += estimate(&part.to_string());
                }
            }
        }
        Some(other) => tokens += estimate(&other.to_string()),
        None => {}
    }
    if let Some(calls) = message.get("tool_calls") {
        tokens += estimate(&calls.to_string());
    }
    tokens
}

/// Shrink one message to about `target` tokens, keeping system prompts from the
/// front and ordinary turns from the end (where the instruction usually is).
/// Returns whether anything changed.
fn trim_message(message: &mut Value, target: u64) -> bool {
    let chars = target.saturating_sub(4).saturating_mul(4) as usize;
    let head = is_system(message);
    let before = message_tokens(message);
    match message.get_mut("content") {
        Some(Value::String(text)) => {
            let clipped = clip(text, chars, head);
            if clipped != *text {
                *text = clipped;
            }
        }
        Some(Value::Array(parts)) => {
            let mut budget = chars;
            // Keep the first part (usually the question) and trim later ones.
            for part in parts.iter_mut().skip(1) {
                let Some(Value::String(text)) = part.get_mut("text") else {
                    continue;
                };
                let clipped = clip(text, budget.min(text.chars().count()), false);
                budget = budget.saturating_sub(clipped.chars().count());
                *text = clipped;
            }
        }
        _ => {}
    }
    message_tokens(message) < before
}

fn clip(text: &str, max_chars: usize, keep_head: bool) -> String {
    let count = text.chars().count();
    if count <= max_chars {
        return text.to_owned();
    }
    if keep_head {
        let mut out: String = text.chars().take(max_chars).collect();
        out.push('\n');
        out.push_str(TRUNCATION_NOTE);
        out
    } else {
        let mut out = String::from(TRUNCATION_NOTE);
        out.push('\n');
        out.extend(text.chars().skip(count - max_chars));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(body: &Value) -> u64 {
        body["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(message_tokens)
            .sum()
    }

    #[test]
    fn no_field_is_untouched() {
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        let before = body.clone();
        assert_eq!(apply(&mut body), None);
        assert_eq!(body, before);
    }

    #[test]
    fn small_requests_are_untouched_but_the_field_is_removed() {
        let mut body = json!({
            "max_context": 8192,
            "messages": [{"role": "user", "content": "2 + 2"}]
        });
        let trim = apply(&mut body).unwrap();
        assert_eq!(trim.dropped_messages, 0);
        assert_eq!(trim.dropped_tokens, 0);
        assert!(body.get("max_context").is_none());
        assert_eq!(body["messages"][0]["content"], "2 + 2");
    }

    #[test]
    fn oldest_turns_are_dropped_first() {
        let mut messages = vec![json!({"role": "system", "content": "be brief"})];
        for index in 0..40 {
            messages.push(json!({
                "role": if index % 2 == 0 { "user" } else { "assistant" },
                "content": "x".repeat(400)
            }));
        }
        messages.push(json!({"role": "user", "content": "the real question"}));
        let mut body = json!({"max_context": 1200, "messages": messages});

        let trim = apply(&mut body).unwrap();
        assert!(trim.changed(), "{trim:?}");
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(
            body["messages"].as_array().unwrap().last().unwrap()["content"],
            "the real question"
        );
        let after = tokens(&body) + estimate(&body["tools"].to_string());
        assert!(after <= 1200, "still {after} tokens: {body}");
    }

    #[test]
    fn tool_calls_and_their_results_are_dropped_together() {
        let mut messages = vec![json!({"role": "system", "content": "s"})];
        for _ in 0..6 {
            messages.push(json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{"id": "c", "type": "function", "function": {"name": "t", "arguments": "{}"}}]
            }));
            messages.push(json!({"role": "tool", "tool_call_id": "c", "content": "x".repeat(2000)}));
        }
        messages.push(json!({"role": "user", "content": "final"}));
        let mut body = json!({"max_context": 3000, "messages": messages});

        apply(&mut body).unwrap();
        let list = body["messages"].as_array().unwrap();
        let tools = list.iter().filter(|m| m["role"] == "tool").count();
        let calls = list.iter().filter(|m| m.get("tool_calls").is_some()).count();
        assert_eq!(tools, calls, "orphaned tool results: {body}");
        assert_eq!(list.last().unwrap()["content"], "final");
    }

    #[test]
    fn an_oversized_single_message_is_trimmed() {
        let mut body = json!({
            "max_context": 600,
            "messages": [
                {"role": "system", "content": "rules"},
                {"role": "user", "content": format!("{} END", "word ".repeat(4000))}
            ]
        });
        let trim = apply(&mut body).unwrap();
        assert!(trim.changed(), "{trim:?}");
        let content = body["messages"][1]["content"].as_str().unwrap();
        assert!(content.starts_with(TRUNCATION_NOTE), "{content}");
        assert!(content.ends_with("END"), "{content}");
        assert!(trim.after_tokens <= 600, "{trim:?}");
    }

    #[test]
    fn max_tokens_is_clamped_to_the_room_left() {
        let mut body = json!({
            "max_context": 500,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": "x".repeat(800 * 4)}]
        });
        apply(&mut body).unwrap();
        let max_tokens = body["max_tokens"].as_u64().unwrap();
        let used = tokens(&body);
        assert!(max_tokens + used <= 500, "max {max_tokens}, used {used}");
        assert!(max_tokens >= 1);
    }

    #[test]
    fn non_numeric_or_zero_limits_are_ignored() {
        for limit in [json!("later"), json!(0), json!(null)] {
            let mut body = json!({"max_context": limit, "messages": [{"role": "user", "content": "hi"}]});
            assert_eq!(apply(&mut body), None);
            assert!(body.get("max_context").is_none());
        }
    }

    #[test]
    fn tool_definitions_count_toward_the_prompt() {
        let mut body = json!({
            "max_context": 8000,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "function": {"name": "t", "description": "d".repeat(4000)}}]
        });
        let trim = apply(&mut body).unwrap();
        assert!(trim.before_tokens >= 1000, "{trim:?}");
        assert_eq!(body["messages"][0]["content"], "hi");
    }
}

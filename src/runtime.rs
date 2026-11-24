//! Runtime facts every prompt needs: the actual date and time, and which model
//! is serving the request.
//!
//! A model without a clock guesses the year from its training data, which makes
//! it call 2026 "the future" and answer time-sensitive questions wrong. The
//! router knows both facts, so it prepends them to the conversation as a system
//! message before each provider attempt (the model name changes per attempt, so
//! the text has to be built there).

use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

/// Marks the injected block, so it is never added twice.
const MARKER: &str = "[runtime]";

/// Prompt room kept for the block when trimming to `max_context`.
pub const RESERVE_TOKENS: u64 = 220;

const WEEKDAYS: [&str; 7] = [
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
];

/// The block injected into a prompt. `model` is the one this attempt sends.
pub fn block(provider: &str, model: &str) -> String {
    block_at(provider, model, SystemTime::now())
}

fn block_at(provider: &str, model: &str, now: SystemTime) -> String {
    format!(
        "{MARKER} Environment facts for this conversation, which you must treat \
         as current and authoritative:\n\
         - Current date and time: {} UTC\n\
         - Model answering this request: {provider}/{model}\n\
         - Requests are routed through a local model router with automatic \
         failover; that is why the model may differ from the one requested.\n\
         Use the date above for anything time-sensitive (the current year, \
         recent events, versions, deadlines, ages). Do not assume an earlier \
         year, and do not call the date above hypothetical or in the future.",
        timestamp(now)
    )
}

/// Prepend the runtime block to an OpenAI-shaped chat body. Returns whether the
/// body changed; bodies without a `messages` array are left alone (embeddings and
/// other non-chat requests).
pub fn inject(body: &mut Value, provider: &str, model: &str) -> bool {
    let Some(object) = body.as_object_mut() else {
        return false;
    };
    let Some(Value::Array(messages)) = object.get_mut("messages") else {
        return false;
    };
    if messages.iter().any(|message| {
        message
            .get("content")
            .is_some_and(|content| content.to_string().contains(MARKER))
    }) {
        return false;
    }

    let text = block(provider, model);
    match messages.first_mut() {
        Some(first) if is_system(first) => match first.get("content") {
            Some(Value::String(_)) => {
                if let Some(Value::String(content)) = first.get_mut("content") {
                    *content = format!("{text}\n\n{content}");
                }
            }
            Some(Value::Array(_)) => {
                if let Some(Value::Array(parts)) = first.get_mut("content") {
                    parts.insert(0, json!({"type": "text", "text": text}));
                }
            }
            // A system message without content is useless anyway.
            _ => first["content"] = Value::String(text),
        },
        _ => messages.insert(0, json!({"role": "system", "content": text})),
    }
    true
}

fn is_system(message: &Value) -> bool {
    matches!(
        message.get("role").and_then(Value::as_str),
        Some("system") | Some("developer")
    )
}

/// `2026-11-23 15:53:27 Monday`
fn timestamp(now: SystemTime) -> String {
    let secs = now
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let day_secs = secs.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    let weekday = WEEKDAYS[(days.rem_euclid(7)) as usize];
    format!(
        "{year:04}-{month:02}-{day:02} {:02}:{:02}:{:02} {weekday}",
        day_secs / 3600,
        day_secs % 3600 / 60,
        day_secs % 60
    )
}

/// Days since the Unix epoch to a civil (year, month, day), Howard Hinnant's
/// `civil_from_days` algorithm.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if month <= 2 { year + 1 } else { year }, month, day)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn at(secs: u64) -> SystemTime {
        UNIX_EPOCH + Duration::from_secs(secs)
    }

    #[test]
    fn formats_known_timestamps() {
        assert_eq!(timestamp(at(0)), "1970-01-01 00:00:00 Thursday");
        // 2026-11-23 15:53:27 UTC was a Monday.
        assert_eq!(timestamp(at(1_795_449_207)), "2026-11-23 15:53:27 Monday");
        assert_eq!(timestamp(at(1_767_225_599)), "2025-12-31 23:59:59 Wednesday");
        assert_eq!(timestamp(at(1_709_164_800)), "2024-02-29 00:00:00 Thursday");
    }

    #[test]
    fn prepends_a_system_message_when_there_is_none() {
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(inject(&mut body, "p", "m"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        let text = messages[0]["content"].as_str().unwrap();
        assert!(text.contains(MARKER));
        assert!(text.contains("p/m"));
        assert!(text.contains("UTC"));
        assert_eq!(messages[1]["content"], "hi");
    }

    #[test]
    fn extends_an_existing_system_prompt() {
        let mut body = json!({"messages": [
            {"role": "system", "content": "you are a writer"},
            {"role": "user", "content": "go"}
        ]});
        assert!(inject(&mut body, "p", "m"));
        let messages = body["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        let text = messages[0]["content"].as_str().unwrap();
        assert!(text.starts_with(MARKER));
        assert!(text.ends_with("you are a writer"));
    }

    #[test]
    fn is_not_added_twice() {
        let mut body = json!({"messages": [{"role": "user", "content": "hi"}]});
        assert!(inject(&mut body, "p", "m"));
        assert!(!inject(&mut body, "p", "m"));
        assert_eq!(body["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn leaves_non_chat_bodies_untouched() {
        let mut body = json!({"input": "hello"});
        assert!(!inject(&mut body, "p", "m"));
        assert_eq!(body, json!({"input": "hello"}));
    }
}

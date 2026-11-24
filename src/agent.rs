//! The web tool loop behind `"web": true` chat requests.
//!
//! One request becomes a small agent run: the model may ask for `web_search`
//! or `web_fetch`, the router executes those itself (so every provider in the
//! improvement layers' fan-out gets true tool calling), the results go back as
//! `tool` messages, and the final answer is returned. Tool calls the router does
//! not own (a harness's own tools) end the loop untouched and are handed back to
//! the client.
//!
//! Nothing is injected into the conversation: the tool descriptions are the only
//! guidance, so an agent request still sees exactly what its harness sent.

use futures_util::future::join_all;
use serde_json::{Map, Value, json};

use crate::router::Router;
use crate::router::RouterError;
use crate::web::WebError;

/// How many times the model may call web tools before it has to answer.
pub const MAX_TOOL_ROUNDS: usize = 2;
/// Tool results are pasted into the conversation, so keep each one small.
const TOOL_CONTENT_CHARS: usize = 8_000;

const WEB_SEARCH: &str = "web_search";
const WEB_FETCH: &str = "web_fetch";

#[derive(Debug, Default)]
pub struct WebTrace {
    pub searches: Vec<SearchTrace>,
    pub fetches: Vec<FetchTrace>,
}

#[derive(Debug)]
pub struct SearchTrace {
    pub query: String,
    pub provider: String,
    pub results: Vec<HitTrace>,
}

#[derive(Debug)]
pub struct HitTrace {
    pub title: String,
    pub url: String,
}

#[derive(Debug)]
pub struct FetchTrace {
    pub url: String,
    pub provider: String,
    pub title: String,
    pub chars: usize,
    pub truncated: bool,
}

impl WebTrace {
    /// What the client gets to show next to the answer.
    pub fn to_json(&self) -> Value {
        json!({
            "searches": self
                .searches
                .iter()
                .map(|search| json!({
                    "query": search.query,
                    "provider": search.provider,
                    "results": search
                        .results
                        .iter()
                        .map(|hit| json!({"title": hit.title, "url": hit.url}))
                        .collect::<Vec<_>>(),
                }))
                .collect::<Vec<_>>(),
            "fetches": self
                .fetches
                .iter()
                .map(|fetch| json!({
                    "url": fetch.url,
                    "provider": fetch.provider,
                    "title": fetch.title,
                    "chars": fetch.chars,
                    "truncated": fetch.truncated,
                }))
                .collect::<Vec<_>>(),
        })
    }

    pub fn is_empty(&self) -> bool {
        self.searches.is_empty() && self.fetches.is_empty()
    }
}

#[derive(Debug)]
pub struct AgentOutcome {
    /// The final assistant message, in OpenAI's shape.
    pub message: Value,
    pub provider: String,
    pub model: String,
    pub trace: WebTrace,
    /// Model calls spent, tool rounds included.
    pub calls: usize,
    /// Tool calls executed.
    pub tool_calls: usize,
}

/// Run the loop. A failed search or fetch is reported back to the model as the
/// tool's result rather than failing the request: it can retry, refine or answer
/// without it.
pub async fn run(router: &Router, body: &Value) -> Result<AgentOutcome, RouterError> {
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let client_tools: Vec<Value> = body
        .get("tools")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let client_names: Vec<String> = client_tools
        .iter()
        .filter_map(|tool| tool.get("function")?.get("name")?.as_str().map(str::to_owned))
        .collect();

    // Only tools the router itself added are executed here.
    let mut own: Vec<&'static str> = Vec::new();
    let mut tools = client_tools.clone();
    if router.config().has_search() && !client_names.iter().any(|name| name == WEB_SEARCH) {
        tools.push(web_search_tool());
        own.push(WEB_SEARCH);
    }
    if router.config().has_fetch() && !client_names.iter().any(|name| name == WEB_FETCH) {
        tools.push(web_fetch_tool());
        own.push(WEB_FETCH);
    }

    let mut trace = WebTrace::default();
    let mut calls = 0usize;
    let mut executed = 0usize;
    let mut conversation = messages;

    for round in 0..=MAX_TOOL_ROUNDS {
        // The last round has no web tools, so the model has to answer.
        let round_tools: Vec<Value> = if round < MAX_TOOL_ROUNDS {
            tools.clone()
        } else {
            client_tools.clone()
        };
        let mut request = body.clone();
        let object = request
            .as_object_mut()
            .expect("checked by the caller: the body is an object");
        object.insert("messages".to_owned(), Value::Array(conversation.clone()));
        object.insert("stream".to_owned(), Value::Bool(false));
        object.remove("stream_options");
        object.remove("web");
        if round_tools.is_empty() {
            object.remove("tools");
            object.remove("tool_choice");
        } else {
            object.insert("tools".to_owned(), Value::Array(round_tools));
        }

        let completion = match router.complete(&request).await {
            Ok(completion) => completion,
            // A provider that rejects tool definitions still owes the user an
            // answer: retry once without the web tools instead of failing.
            Err(err) if round == 0 && !own.is_empty() && is_rejected_request(&err) => {
                tracing::warn!(error = %err, "provider rejected the web tools, answering without them");
                let mut retry = request.clone();
                if let Some(object) = retry.as_object_mut() {
                    if client_tools.is_empty() {
                        object.remove("tools");
                        object.remove("tool_choice");
                    } else {
                        object.insert("tools".to_owned(), Value::Array(client_tools.clone()));
                    }
                }
                let completion = router.complete(&retry).await?;
                calls += 1;
                let message = message_of(&completion);
                return Ok(finish(completion, message, trace, calls, 0));
            }
            Err(err) => return Err(err),
        };
        calls += 1;
        let message = message_of(&completion);

        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        if tool_calls.is_empty() {
            return Ok(finish(completion, message, trace, calls, executed));
        }

        let owned: Vec<&Value> = tool_calls
            .iter()
            .filter(|call| {
                call.get("function")
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .is_some_and(|name| own.contains(&name))
            })
            .collect();
        // A harness's own tools (or a web tool the client declared itself) are
        // the client's to run: hand the message back untouched.
        if owned.len() != tool_calls.len() || round == MAX_TOOL_ROUNDS {
            return Ok(finish(completion, message, trace, calls, executed));
        }

        let results = join_all(owned.iter().map(|call| execute(router, call))).await;
        executed += results.len();

        conversation.push(message.clone());
        for (call, (content, note)) in owned.iter().zip(results) {
            match note {
                Some(ToolNote::Search(search)) => trace.searches.push(search),
                Some(ToolNote::Fetch(fetch)) => trace.fetches.push(fetch),
                None => {}
            }
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            conversation.push(json!({
                "role": "tool",
                "tool_call_id": id,
                "content": content,
            }));
        }
        tracing::info!(
            tools = executed,
            provider = %completion.provider,
            "web tools ran, asking the model again"
        );
    }

    // Unreachable: the last round always returns above.
    Err(RouterError::BadRequest(
        "the web tool loop did not terminate".to_owned(),
    ))
}

/// The assistant message of a completion, or an empty one.
fn message_of(completion: &crate::router::Completion) -> Value {
    completion.body["choices"]
        .as_array()
        .and_then(|choices| choices.first())
        .and_then(|choice| choice.get("message"))
        .cloned()
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({"role": "assistant", "content": ""}))
}

/// Every attempt was refused as a bad request (typically "tools are not
/// supported"), so the caller can retry with a simpler body.
fn is_rejected_request(err: &RouterError) -> bool {
    match err {
        RouterError::BadRequest(_) => true,
        RouterError::AllFailed(failures) => {
            !failures.is_empty() && failures.iter().all(|failure| failure.error.contains("400"))
        }
        RouterError::Plan(_) => false,
    }
}

fn finish(
    completion: crate::router::Completion,
    message: Value,
    trace: WebTrace,
    calls: usize,
    tool_calls: usize,
) -> AgentOutcome {
    AgentOutcome {
        message,
        provider: completion.provider,
        model: completion.model,
        trace,
        calls,
        tool_calls,
    }
}

/// What a successful tool call added to the client-visible trace.
enum ToolNote {
    Search(SearchTrace),
    Fetch(FetchTrace),
}

/// Run one tool call and render its result for the model.
async fn execute(router: &Router, call: &Value) -> (String, Option<ToolNote>) {
    let function = call.get("function").cloned().unwrap_or(Value::Null);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let arguments = parse_arguments(function.get("arguments"));

    match name.as_str() {
        WEB_SEARCH => {
            let query = arguments
                .get("query")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let limit = arguments
                .get("limit")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            match router.web_search(&query, limit).await {
                Ok(outcome) => {
                    let note = SearchTrace {
                        query,
                        provider: outcome.provider,
                        results: outcome
                            .results
                            .iter()
                            .take(5)
                            .map(|hit| HitTrace {
                                title: hit.title.clone(),
                                url: hit.url.clone(),
                            })
                            .collect(),
                    };
                    (clip(&outcome.text), Some(ToolNote::Search(note)))
                }
                Err(err) => (tool_error(WEB_SEARCH, &err), None),
            }
        }
        WEB_FETCH => {
            let url = arguments
                .get("url")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_owned();
            match router.web_fetch(&url).await {
                Ok(outcome) => {
                    let note = FetchTrace {
                        url,
                        provider: outcome.provider,
                        title: outcome.title.clone(),
                        chars: outcome.text.len(),
                        truncated: outcome.truncated,
                    };
                    (clip(&outcome.text), Some(ToolNote::Fetch(note)))
                }
                Err(err) => (tool_error(WEB_FETCH, &err), None),
            }
        }
        other => (format!("unknown tool `{other}`"), None),
    }
}

fn tool_error(tool: &str, err: &WebError) -> String {
    tracing::warn!(tool, error = %err, "web tool failed");
    format!("{tool} failed: {err}")
}

fn clip(text: &str) -> String {
    if text.chars().count() <= TOOL_CONTENT_CHARS {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(TOOL_CONTENT_CHARS).collect();
    out.push_str("\n… (truncated)");
    out
}

/// Arguments arrive as a JSON string; a model that emits invalid JSON still
/// gets a useful answer instead of a dead request.
fn parse_arguments(raw: Option<&Value>) -> Map<String, Value> {
    match raw {
        Some(Value::Object(map)) => map.clone(),
        Some(Value::String(text)) => serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| value.as_object().cloned())
            .unwrap_or_default(),
        _ => Map::new(),
    }
}

fn web_search_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": WEB_SEARCH,
            "description": "Search the web. Use it for anything that may have changed since your \
                            training data: news, prices, releases, versions, documentation, \
                            current events, or facts you are unsure about.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "limit": {
                        "type": "integer",
                        "description": "How many results to return (1-10, default 6)."
                    }
                },
                "required": ["query"]
            }
        }
    })
}

fn web_fetch_tool() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": WEB_FETCH,
            "description": "Read one web page as clean text. Use it on a URL you found with \
                            web_search when the snippet is not enough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The http(s) URL to read."}
                },
                "required": ["url"]
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use axum::extract::State;
    use axum::response::{IntoResponse, Response};
    use axum::routing::{get, post};

    use super::*;
    use crate::config::Config;
    use crate::router::Options;

    const DDG_HTML: &str = r#"
<div class="result">
  <a rel="nofollow" class="result__a" href="https://example.com/axum">Axum docs</a>
  <a class="result__snippet">Routing and extracting in axum.</a>
</div>"#;

    #[derive(Clone, Default)]
    struct Mock {
        seen: Arc<Mutex<Vec<Value>>>,
    }

    async fn complete(State(mock): State<Mock>, body: String) -> Response {
        let value: Value = serde_json::from_str(&body).unwrap();
        mock.seen.lock().unwrap().push(value.clone());
        let answered = value["messages"]
            .as_array()
            .is_some_and(|messages| messages.iter().any(|m| m["role"] == "tool"));
        let message = if answered {
            json!({"role": "assistant", "content": "Axum routes with Router::new()."})
        } else {
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "web_search", "arguments": "{\"query\":\"axum routing\"}"}
                }]
            })
        };
        axum::Json(json!({
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}]
        }))
        .into_response()
    }

    async fn search_page() -> &'static str {
        DDG_HTML
    }

    async fn spawn_router() -> (String, Mock) {
        let mock = Mock::default();
        let app = axum::Router::new()
            .route("/v1/chat/completions", post(complete))
            .route("/html/", get(search_page))
            .with_state(mock.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        (format!("http://{addr}"), mock)
    }

    #[tokio::test]
    async fn a_search_call_comes_back_as_a_tool_message() {
        let (base, mock) = spawn_router().await;
        let config = Config::parse(&format!(
            "[providers]\nopenai,{base}/v1,mock,key:k\nduckduckgo,{base},ddg\n[models]\nmock/m\n"
        ))
        .unwrap();
        let router = Router::new(
            config,
            Options {
                attempt_timeout: Duration::from_secs(5),
                model_timeout: Duration::from_secs(1),
            },
        )
        .unwrap();

        let outcome = run(
            &router,
            &json!({
                "model": "auto",
                "web": true,
                "stream": true,
                "messages": [{"role": "user", "content": "how does axum route?"}],
            }),
        )
        .await
        .unwrap();

        assert_eq!(outcome.calls, 2, "one tool round, then the answer");
        assert_eq!(outcome.tool_calls, 1);
        assert_eq!(outcome.message["content"], "Axum routes with Router::new().");
        assert_eq!(outcome.trace.searches.len(), 1);
        assert_eq!(outcome.trace.searches[0].query, "axum routing");
        assert_eq!(outcome.trace.searches[0].provider, "ddg");
        assert_eq!(outcome.trace.searches[0].results[0].url, "https://example.com/axum");

        let seen = mock.seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        // The second call carries the assistant's tool call and the result.
        let messages = seen[1]["messages"].as_array().unwrap();
        assert!(messages.iter().any(|m| m["role"] == "assistant" && m["tool_calls"].is_array()));
        let tool = messages.iter().find(|m| m["role"] == "tool").unwrap();
        assert_eq!(tool["tool_call_id"], "call_1");
        assert!(tool["content"].as_str().unwrap().contains("https://example.com/axum"));
        // Web tools are advertised, and `web` never reaches the provider.
        assert_eq!(seen[0]["tools"][0]["function"]["name"], "web_search");
        assert!(seen[0].get("web").is_none());
        assert_eq!(seen[0]["stream"], false);
    }

    /// A provider that refuses requests carrying `tools`.
    async fn refuse_tools(body: String) -> Response {
        let value: Value = serde_json::from_str(&body).unwrap();
        if value.get("tools").is_some() {
            return (
                axum::http::StatusCode::BAD_REQUEST,
                axum::Json(json!({"error": {"message": "tools are not supported"}})),
            )
                .into_response();
        }
        axum::Json(json!({
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "plain answer"}}]
        }))
        .into_response()
    }

    #[tokio::test]
    async fn providers_that_reject_tools_still_answer() {
        let app = axum::Router::new().route("/v1/chat/completions", post(refuse_tools));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let base = format!("http://{addr}");
        let config = Config::parse(&format!(
            "[providers]\nopenai,{base}/v1,mock,key:k\nduckduckgo,{base},ddg\n[models]\nmock/m\n"
        ))
        .unwrap();
        let router = Router::new(
            config,
            Options {
                attempt_timeout: Duration::from_secs(5),
                model_timeout: Duration::from_secs(1),
            },
        )
        .unwrap();

        let outcome = run(
            &router,
            &json!({
                "model": "auto",
                "web": true,
                "messages": [{"role": "user", "content": "hi"}],
            }),
        )
        .await
        .unwrap();

        assert_eq!(outcome.message["content"], "plain answer");
        assert_eq!(outcome.calls, 1);
        assert_eq!(outcome.tool_calls, 0);
    }

    #[tokio::test]
    async fn client_tools_end_the_loop_untouched() {
        let (base, _mock) = spawn_router().await;
        let config = Config::parse(&format!(
            "[providers]\nopenai,{base}/v1,mock,key:k\n[models]\nmock/m\n"
        ))
        .unwrap();
        let router = Router::new(
            config,
            Options {
                attempt_timeout: Duration::from_secs(5),
                model_timeout: Duration::from_secs(1),
            },
        )
        .unwrap();
        // The mock always asks for `read_file`, a tool the router does not own.
        let outcome = run(
            &router,
            &json!({
                "model": "auto",
                "web": true,
                "messages": [{"role": "user", "content": "read a.txt"}],
                "tools": [{"type": "function", "function": {"name": "read_file"}}],
            }),
        )
        .await
        .unwrap();
        assert_eq!(outcome.calls, 1, "no second call: the harness runs it");
        assert_eq!(outcome.tool_calls, 0);
        assert!(outcome.message["tool_calls"].is_array());
        assert!(outcome.trace.is_empty());
    }

    #[test]
    fn tool_arguments_are_lenient() {
        assert_eq!(
            parse_arguments(Some(&json!("{\"query\":\"x\"}")))["query"],
            "x"
        );
        assert_eq!(parse_arguments(Some(&json!({"query": "y"})))["query"], "y");
        assert!(parse_arguments(Some(&json!("{oops"))).is_empty());
        assert!(parse_arguments(None).is_empty());
    }

    #[test]
    fn tool_content_is_clipped() {
        let long = "a".repeat(TOOL_CONTENT_CHARS + 10);
        let clipped = clip(&long);
        assert!(clipped.ends_with("(truncated)"));
        assert_eq!(clip("short"), "short");
    }

    #[test]
    fn trace_becomes_json_for_the_client() {
        let mut trace = WebTrace::default();
        trace.searches.push(SearchTrace {
            query: "axum".into(),
            provider: "ddg".into(),
            results: vec![HitTrace {
                title: "Axum".into(),
                url: "https://docs.rs/axum".into(),
            }],
        });
        trace.fetches.push(FetchTrace {
            url: "https://docs.rs/axum".into(),
            provider: "jina".into(),
            title: "Axum".into(),
            chars: 1200,
            truncated: false,
        });
        let value = trace.to_json();
        assert_eq!(value["searches"][0]["provider"], "ddg");
        assert_eq!(value["searches"][0]["results"][0]["url"], "https://docs.rs/axum");
        assert_eq!(value["fetches"][0]["chars"], 1200);
        assert!(!trace.is_empty());
    }
}

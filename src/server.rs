//! The HTTP surface: an OpenAI-compatible endpoint that forwards each request
//! to whichever configured provider accepts it first, plus the web tools.
//!
//! * `POST /v1/chat/completions` with `"web": true` runs [`crate::agent`]: the
//!   model may call `web_search`/`web_fetch`, the router executes them, and the
//!   final answer comes back (as a synthesized stream when `stream` is set).
//! * `POST /v1/web_search` and `POST /v1/web_fetch` expose the same tools
//!   directly, for clients that prefer to drive the loop themselves.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::extract::{Request, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{any, get, post};
use axum::{Json, Router as HttpRouter};
use bytes::Bytes;
use serde_json::{Value, json};

use crate::agent::{self, AgentOutcome};
use crate::config::PlanError;
use crate::router::{ForwardRequest, Router, RouterError, Upstream};
use crate::web::WebError;

/// Requests are buffered so a failed attempt can be replayed on the next provider.
const MAX_BODY_BYTES: usize = 32 * 1024 * 1024;
/// Roughly how much text each synthesized SSE chunk carries.
const STREAM_CHUNK_CHARS: usize = 64;

#[derive(Clone)]
pub struct AppState {
    pub router: Arc<Router>,
    /// When set, clients must send `Authorization: Bearer <token>`.
    pub token: Option<Arc<str>>,
}

pub fn app(state: AppState) -> HttpRouter {
    HttpRouter::new()
        .route("/health", get(health))
        // OpenAI has no health endpoint, but clients (and people with curl) ask
        // for `/v1/health` anyway. Answering it here keeps those probes from
        // being forwarded to every provider as if they were chat requests.
        .route("/v1/health", get(health))
        .route("/v1/web_search", post(web_search))
        .route("/v1/web_fetch", post(web_fetch))
        .route("/v1/{*path}", any(proxy))
        .with_state(state)
}

async fn health(State(state): State<AppState>) -> impl IntoResponse {
    let config = state.router.config();
    Json(json!({
        "status": "ok",
        "providers": config.providers.iter().map(|p| json!({
            "shortname": p.shortname,
            "base_url": p.base_url,
            "api": p.api.as_str(),
            "role": p.api.role().as_str(),
            "has_key": p.api_key().is_some(),
        })).collect::<Vec<_>>(),
        "models": config.models.iter().map(|m| m.raw.clone()).collect::<Vec<_>>(),
        // What `"web": true` can do, so a client knows whether to ask for it.
        "web": {
            "search": config.search_providers().map(|(_, p)| json!({
                "shortname": p.shortname,
                "api": p.api.as_str(),
            })).collect::<Vec<_>>(),
            "fetch": config.fetch_providers().map(|(_, p)| json!({
                "shortname": p.shortname,
                "api": p.api.as_str(),
            })).collect::<Vec<_>>(),
        },
    }))
}

async fn proxy(State(state): State<AppState>, request: Request) -> Response {
    if let Some(response) = unauthorized(&state, request.headers()) {
        return response;
    }
    if request.method() == Method::GET && request.uri().path() == "/v1/models" {
        return models(&state).into_response();
    }

    let (parts, body) = request.into_parts();
    let path_and_query = parts
        .uri
        .path_and_query()
        .map(|p| p.as_str().to_owned())
        .unwrap_or_else(|| "/".to_owned());
    let body = match axum::body::to_bytes(body, MAX_BODY_BYTES).await {
        Ok(body) => body,
        Err(_) => {
            return error_response(
                StatusCode::PAYLOAD_TOO_LARGE,
                "payload_too_large",
                &format!("request body must be valid and at most {MAX_BODY_BYTES} bytes"),
            );
        }
    };

    // `"web": true` turns a chat request into a small agent run.
    if parts.method == Method::POST
        && parts.uri.path() == "/v1/chat/completions"
        && let Ok(Value::Object(object)) = serde_json::from_slice::<Value>(&body)
        && object.get("web").and_then(Value::as_bool) == Some(true)
    {
        return run_agent(&state, Value::Object(object)).await;
    }

    let forward = ForwardRequest {
        method: parts.method,
        path_and_query,
        headers: parts.headers,
        body,
    };
    match state.router.proxy(forward).await {
        Ok(upstream) => upstream.into_response(),
        Err(err) => err.into_response(),
    }
}

// ------------------------------------------------------------------ web tools

async fn web_search(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if let Some(response) = unauthorized(&state, &headers) {
        return response;
    }
    let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let query = request
        .get("query")
        .or_else(|| request.get("q"))
        .and_then(Value::as_str)
        .unwrap_or_default();
    let limit = request.get("limit").and_then(Value::as_u64).unwrap_or(0) as usize;

    match state.router.web_search(query, limit).await {
        Ok(outcome) => Json(json!({
            "query": query,
            "provider": outcome.provider,
            "results": outcome.results.iter().map(|hit| json!({
                "title": hit.title,
                "url": hit.url,
                "snippet": hit.snippet,
            })).collect::<Vec<_>>(),
            "text": outcome.text,
        }))
        .into_response(),
        Err(err) => err.into_response(),
    }
}

async fn web_fetch(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    if let Some(response) = unauthorized(&state, &headers) {
        return response;
    }
    let request: Value = serde_json::from_slice(&body).unwrap_or(Value::Null);
    let url = request
        .get("url")
        .or_else(|| request.get("href"))
        .and_then(Value::as_str)
        .unwrap_or_default();

    match state.router.web_fetch(url).await {
        Ok(outcome) => Json(json!({
            "url": outcome.url,
            "provider": outcome.provider,
            "title": outcome.title,
            "text": outcome.text,
            "truncated": outcome.truncated,
        }))
        .into_response(),
        Err(err) => err.into_response(),
    }
}

async fn run_agent(state: &AppState, request: Value) -> Response {
    let stream = request
        .get("stream")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    match agent::run(&state.router, &request).await {
        Ok(outcome) => {
            tracing::info!(
                provider = %outcome.provider,
                model = %outcome.model,
                calls = outcome.calls,
                tool_calls = outcome.tool_calls,
                searches = outcome.trace.searches.len(),
                fetches = outcome.trace.fetches.len(),
                stream,
                "web agent answered"
            );
            if stream {
                agent_stream_response(outcome)
            } else {
                agent_json_response(outcome)
            }
        }
        Err(err) => err.into_response(),
    }
}

fn agent_json_response(outcome: AgentOutcome) -> Response {
    let mut response = Json(json!({
        "id": completion_id(),
        "object": "chat.completion",
        "created": now_secs(),
        "model": outcome.model,
        "choices": [{
            "index": 0,
            "message": outcome.message,
            "finish_reason": finish_reason(&outcome.message),
        }],
        "web": outcome.trace.to_json(),
    }))
    .into_response();
    describe(response.headers_mut(), &outcome);
    response
}

/// A synthesized OpenAI SSE stream: the web trace as a choice-less chunk, then
/// the final answer. The loop needs buffered answers, so tokens are chunked
/// after the fact instead of streamed live.
fn agent_stream_response(outcome: AgentOutcome) -> Response {
    let id = completion_id();
    let created = now_secs();
    let model = outcome.model.clone();
    let mut frames: Vec<Bytes> = Vec::new();

    frames.push(sse_frame(&json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [],
        "web": outcome.trace.to_json(),
    })));
    frames.push(chunk_frame(
        &id,
        created,
        &model,
        json!({"role": "assistant", "content": ""}),
        None,
    ));

    if let Some(text) = outcome.message.get("content").and_then(Value::as_str) {
        for part in text_chunks(text, STREAM_CHUNK_CHARS) {
            frames.push(chunk_frame(
                &id,
                created,
                &model,
                json!({"content": part}),
                None,
            ));
        }
    }
    if let Some(calls) = outcome.message.get("tool_calls").and_then(Value::as_array) {
        for (index, call) in calls.iter().enumerate() {
            let mut delta = json!({
                "index": index,
                "type": "function",
                "function": call.get("function").cloned().unwrap_or(Value::Null),
            });
            if let Some(call_id) = call.get("id") {
                delta["id"] = call_id.clone();
            }
            frames.push(chunk_frame(
                &id,
                created,
                &model,
                json!({"tool_calls": [delta]}),
                None,
            ));
        }
    }
    frames.push(chunk_frame(
        &id,
        created,
        &model,
        json!({}),
        Some(finish_reason(&outcome.message)),
    ));
    frames.push(Bytes::from_static(b"data: [DONE]\n\n"));

    let body = Body::from_stream(async_stream::stream! {
        for frame in frames {
            yield Ok::<Bytes, std::convert::Infallible>(frame);
        }
    });
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/event-stream"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    describe(response.headers_mut(), &outcome);
    response
}

fn describe(headers: &mut HeaderMap, outcome: &AgentOutcome) {
    insert_header(headers, "x-router-provider", &outcome.provider);
    insert_header(headers, "x-router-model", &outcome.model);
    insert_header(headers, "x-router-web-searches", &outcome.trace.searches.len().to_string());
    insert_header(headers, "x-router-web-fetches", &outcome.trace.fetches.len().to_string());
}

fn chunk_frame(id: &str, created: u64, model: &str, delta: Value, finish: Option<&str>) -> Bytes {
    sse_frame(&json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }))
}

fn sse_frame(value: &Value) -> Bytes {
    Bytes::from(format!("data: {value}\n\n"))
}

fn completion_id() -> String {
    format!("chatcmpl-router-{}", now_secs())
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

fn finish_reason(message: &Value) -> &'static str {
    let has_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| !calls.is_empty());
    if has_calls {
        "tool_calls"
    } else {
        "stop"
    }
}

/// Split text on whitespace into chunks of about `target` characters.
fn text_chunks(text: &str, target: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    for word in text.split_inclusive(char::is_whitespace) {
        current.push_str(word);
        if current.len() >= target {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

fn models(state: &AppState) -> Json<serde_json::Value> {
    Json(json!({
        "object": "list",
        "data": state.router.config().models.iter().map(|m| json!({
            "id": m.raw,
            "object": "model",
            "owned_by": m.provider,
        })).collect::<Vec<_>>(),
    }))
}

/// Check the router's own bearer token, if one is configured.
fn unauthorized(state: &AppState, headers: &HeaderMap) -> Option<Response> {
    let expected = state.token.as_deref()?;
    let provided = headers
        .get(header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
        .unwrap_or_default();
    (provided != expected).then(|| {
        error_response(
            StatusCode::UNAUTHORIZED,
            "unauthorized",
            "missing or invalid router token",
        )
    })
}

fn error_response(status: StatusCode, code: &str, message: &str) -> Response {
    (
        status,
        Json(json!({
            "error": {
                "message": message,
                "type": "router_error",
                "code": code,
            }
        })),
    )
        .into_response()
}

impl IntoResponse for WebError {
    fn into_response(self) -> Response {
        let status = match &self {
            WebError::NoProviders(_) => StatusCode::SERVICE_UNAVAILABLE,
            WebError::EmptyQuery | WebError::BadUrl(_) | WebError::PrivateAddress(_) => {
                StatusCode::BAD_REQUEST
            }
            WebError::AllFailed(_) => StatusCode::BAD_GATEWAY,
        };
        let code = match &self {
            WebError::NoProviders(_) => "no_web_providers",
            WebError::EmptyQuery => "invalid_request",
            WebError::BadUrl(_) => "invalid_url",
            WebError::PrivateAddress(_) => "private_address",
            WebError::AllFailed(_) => "all_web_providers_failed",
        };
        error_response(status, code, &self.to_string())
    }
}

impl IntoResponse for Upstream {
    fn into_response(self) -> Response {
        let mut response = Response::new(self.body);
        *response.status_mut() = self.status;
        if let Ok(value) = HeaderValue::from_str(&self.content_type) {
            response.headers_mut().insert(header::CONTENT_TYPE, value);
        }
        insert_header(response.headers_mut(), "x-router-provider", &self.provider);
        insert_header(response.headers_mut(), "x-router-model", &self.model);
        if let Some(trim) = self.context.filter(|trim| trim.changed()) {
            insert_header(
                response.headers_mut(),
                "x-router-context-dropped",
                &trim.dropped_tokens.to_string(),
            );
            insert_header(
                response.headers_mut(),
                "x-router-context-messages",
                &trim.dropped_messages.to_string(),
            );
        }
        response
    }
}

impl IntoResponse for RouterError {
    fn into_response(self) -> Response {
        match self {
            RouterError::Plan(PlanError::NoModels) => error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "no_models",
                "the router has no models configured",
            ),
            RouterError::Plan(err @ PlanError::UnknownModel { .. }) => {
                error_response(StatusCode::BAD_REQUEST, "unknown_model", &err.to_string())
            }
            RouterError::BadRequest(message) => {
                error_response(StatusCode::BAD_REQUEST, "invalid_request", &message)
            }
            RouterError::AllFailed(failures) => {
                let attempts: Vec<_> = failures
                    .iter()
                    .map(|f| {
                        json!({
                            "provider": f.provider,
                            "model": f.model,
                            "error": f.error,
                        })
                    })
                    .collect();
                (
                    StatusCode::BAD_GATEWAY,
                    Json(json!({
                        "error": {
                            "message": format!("all {} provider attempt(s) failed", failures.len()),
                            "type": "router_error",
                            "code": "all_providers_failed",
                            "attempts": attempts,
                        }
                    })),
                )
                    .into_response()
            }
        }
    }
}

fn insert_header(headers: &mut HeaderMap, name: &'static str, value: &str) {
    if let Ok(value) = HeaderValue::from_str(value) {
        headers.insert(name, value);
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::config::Config;
    use crate::router::Options;

    /// A router whose only provider is unreachable: any request that reaches an
    /// upstream attempt must fail loudly.
    async fn spawn_app() -> String {
        let config = Config::parse(
            "[providers]\nopenai,http://127.0.0.1:1/v1,mock\n[models]\nmock/m\n",
        )
        .unwrap();
        let router = Router::new(
            config,
            Options {
                attempt_timeout: Duration::from_secs(1),
                model_timeout: Duration::from_secs(1),
            },
        )
        .unwrap();
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let app = app(AppState {
            router: Arc::new(router),
            token: None,
        });
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn health_answers_locally_under_v1_too() {
        let base = spawn_app().await;
        let response = reqwest::get(format!("{base}/v1/health")).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body: serde_json::Value = serde_json::from_slice(&response.bytes().await.unwrap()).unwrap();
        assert_eq!(body["status"], "ok");
        assert!(body["providers"].is_array(), "{body}");
        assert!(body["web"]["search"].is_array(), "{body}");
    }

    #[tokio::test]
    async fn web_search_without_providers_is_a_clear_503() {
        let base = spawn_app().await;
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/web_search"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"query":"rust"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let body: serde_json::Value =
            serde_json::from_slice(&response.bytes().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "no_web_providers");
        assert!(body["error"]["message"].as_str().unwrap().contains("duckduckgo"));
    }

    #[tokio::test]
    async fn web_fetch_refuses_local_addresses() {
        let base = spawn_app().await;
        let response = reqwest::Client::new()
            .post(format!("{base}/v1/web_fetch"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(r#"{"url":"http://127.0.0.1:22/"}"#)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let body: serde_json::Value =
            serde_json::from_slice(&response.bytes().await.unwrap()).unwrap();
        assert_eq!(body["error"]["code"], "private_address");
    }

    #[test]
    fn chunks_cover_the_whole_text() {
        let text = "hello brave new world, this is longer";
        let chunks = text_chunks(text, 10);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
    }
}

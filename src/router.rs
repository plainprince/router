//! Failover across the configured providers.
//!
//! The `[models]` list in the config is walked in exactly its written order.
//! An attempt that fails in any way — a non-2xx status, no response headers in
//! time, a dropped connection, an unreadable body — marks that `provider/model`
//! as failed for the configured timeout and starts the walk over from the top
//! of the list, skipping the models still on timeout. When the end of the list
//! is reached, every timeout is cleared and the walk repeats; after two repeats
//! the request fails with every attempt's error.
//!
//! Failure state is per `provider/model` (not per provider), so one dead model
//! never hides its neighbours. It lives in an atomic shared by all requests:
//! while one request restarts its walk, concurrent requests skip the same model
//! immediately instead of waiting on it.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::body::Body;
use axum::http::{HeaderMap, Method, StatusCode, header};
use bytes::Bytes;
use serde_json::Value;
use thiserror::Error;

use crate::anthropic;
use crate::config::{ApiKind, Candidate, Config, PlanError};
use crate::context;
use crate::runtime;

/// How many times the whole candidate list is walked before giving up: the
/// first walk plus two repeats after clearing every timeout.
const WALKS: usize = 3;

/// Headers that must not be relayed to the upstream provider.
const STRIPPED_HEADERS: [&str; 11] = [
    "authorization",
    "accept-encoding",
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
];

#[derive(Debug, Clone, Copy)]
pub struct Options {
    /// How long to wait for a provider's answer (headers, and the body of a
    /// buffered response) before treating it as failed.
    pub attempt_timeout: Duration,
    /// How long a failed `provider/model` is skipped before it may be tried
    /// again.
    pub model_timeout: Duration,
}

pub struct Router {
    config: Config,
    pub(crate) client: reqwest::Client,
    pub(crate) options: Options,
    /// Failure timeout per `provider/model`, created on first use.
    pub(crate) timeouts: Mutex<HashMap<String, Arc<Cooldown>>>,
}

#[derive(Debug, Default)]
pub(crate) struct Cooldown {
    /// Unix milliseconds until which this model is skipped after a failure.
    blocked_until_ms: AtomicU64,
}

impl Cooldown {
    pub(crate) fn is_available(&self, now_ms: u64) -> bool {
        self.blocked_until_ms.load(Ordering::Relaxed) <= now_ms
    }

    pub(crate) fn block(&self, timeout: Duration) {
        let until = now_ms().saturating_add(timeout.as_millis() as u64);
        // Concurrent failures may extend the window; never shorten it.
        self.blocked_until_ms.fetch_max(until, Ordering::Relaxed);
    }

    /// Make the model eligible again (used when the list has been exhausted).
    pub(crate) fn clear(&self) {
        self.blocked_until_ms.store(0, Ordering::Relaxed);
    }
}

/// A request on its way to a provider.
pub struct ForwardRequest {
    pub method: Method,
    /// Path and query exactly as the client sent it, e.g. `/v1/chat/completions`.
    pub path_and_query: String,
    pub headers: HeaderMap,
    pub body: Bytes,
}

/// A buffered completion: the provider's response in the OpenAI dialect.
#[derive(Debug)]
pub struct Completion {
    pub provider: String,
    /// Model name actually sent upstream (provider prefix removed).
    pub model: String,
    pub body: Value,
}

/// A provider's response, in the OpenAI dialect the client asked for.
#[derive(Debug)]
pub struct Upstream {
    pub provider: String,
    /// Model name actually sent upstream (provider prefix removed).
    pub model: String,
    pub status: StatusCode,
    pub content_type: String,
    pub body: Body,
    /// Set when `max_context` removed part of the prompt.
    pub context: Option<context::Trim>,
}

#[derive(Debug, Clone)]
pub struct Failure {
    pub provider: String,
    pub model: String,
    pub error: String,
}

#[derive(Debug, Error)]
pub enum RouterError {
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// The client's request cannot be expressed for any provider.
    #[error("{0}")]
    BadRequest(String),
    #[error("all providers failed ({})", summarize(.0))]
    AllFailed(Vec<Failure>),
}

/// Why one attempt did not produce a response.
enum AttemptError {
    /// This provider could not serve the request; try the next one.
    Skip(Failure),
    /// The request itself is malformed, so no provider will accept it.
    Client(String),
}

impl Router {
    pub fn new(config: Config, options: Options) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            // Fail fast on stalled connections instead of hanging the caller.
            .tcp_nodelay(true)
            .pool_idle_timeout(Duration::from_secs(90))
            .build()?;
        Ok(Self {
            config,
            client,
            options,
            timeouts: Mutex::new(HashMap::new()),
        })
    }

    fn slot_for(&self, key: String) -> Arc<Cooldown> {
        let mut slots = self
            .timeouts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        slots.entry(key).or_default().clone()
    }

    /// The slot for one attempt target. Failures are per model, not per
    /// provider, so the key is `provider/model`; an unconfigured
    /// `provider/model` pair gets its own slot too.
    fn slot(&self, candidate: &Candidate<'_>) -> Arc<Cooldown> {
        self.slot_for(format!("{}/{}", candidate.provider.shortname, candidate.model))
    }

    /// Web tools have no model, so their state is per provider.
    pub(crate) fn web_slot(&self, provider: &crate::config::Provider) -> Arc<Cooldown> {
        self.slot_for(format!("web:{}", provider.shortname))
    }

    fn is_available(&self, candidate: &Candidate<'_>) -> bool {
        self.slot(candidate).is_available(now_ms())
    }

    /// Mark a model as failed; it is skipped until its timeout expires.
    fn fail(&self, candidate: &Candidate<'_>) {
        tracing::debug!(
            provider = %candidate.provider.shortname,
            model = %candidate.model,
            timeout_secs = self.options.model_timeout.as_secs(),
            "model failed, skipping it until its timeout expires"
        );
        self.slot(candidate).block(self.options.model_timeout);
    }

    /// Clear every timeout so the list can be walked again from the top.
    fn clear_timeouts(&self, candidates: &[Candidate<'_>]) {
        for candidate in candidates {
            self.slot(candidate).clear();
        }
    }

    /// The next candidate to try: the first in list order that was not already
    /// tried in this walk and is not on timeout.
    fn next_candidate<'c, 'a>(
        &self,
        candidates: &'c [Candidate<'a>],
        tried: &[bool],
    ) -> Option<(usize, &'c Candidate<'a>)> {
        candidates
            .iter()
            .enumerate()
            .find(|(index, candidate)| !tried[*index] && self.is_available(candidate))
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// One buffered chat completion, with the same failover as [`Self::proxy`].
    ///
    /// The web tool loop and the improvement layers both need the whole answer
    /// before deciding what to do next, so this waits for the body.
    pub async fn complete(&self, body: &Value) -> Result<Completion, RouterError> {
        if !body.is_object() {
            return Err(RouterError::BadRequest(
                "request body must be a JSON object".to_owned(),
            ));
        }
        let mut body = body.clone();
        if let Some(trim) = context::apply(&mut body).filter(|trim| trim.changed()) {
            tracing::info!(
                dropped_messages = trim.dropped_messages,
                before_tokens = trim.before_tokens,
                after_tokens = trim.after_tokens,
                "trimmed the prompt to the requested context window"
            );
        }
        let requested = body.get("model").and_then(Value::as_str).map(str::to_owned);
        let candidates = self.config.candidates(requested.as_deref())?;

        let mut failures = Vec::with_capacity(candidates.len());
        // The list is walked in order, restarting from the top after every
        // failure; a walk ends once every model has been tried or skipped.
        for walk in 0..WALKS {
            if walk > 0 {
                self.clear_timeouts(&candidates);
            }
            let mut tried = vec![false; candidates.len()];
            while let Some((index, candidate)) = self.next_candidate(&candidates, &tried) {
                tried[index] = true;
                match self.complete_one(&body, candidate).await {
                    Ok(completion) => return Ok(completion),
                    Err(AttemptError::Skip(failure)) => {
                        self.fail(candidate);
                        failures.push(failure);
                    }
                    Err(AttemptError::Client(message)) => {
                        return Err(RouterError::BadRequest(message));
                    }
                }
            }
        }
        Err(RouterError::AllFailed(dedupe(failures)))
    }

    async fn complete_one(
        &self,
        body: &Value,
        candidate: &Candidate<'_>,
    ) -> Result<Completion, AttemptError> {
        let attempt_timeout = self.options.attempt_timeout;
        let provider = candidate.provider;
        let failure = |error: String| {
            AttemptError::Skip(Failure {
                provider: provider.shortname.clone(),
                model: candidate.model.clone(),
                error,
            })
        };

        let mut body = body.clone();
        body["model"] = Value::String(candidate.model.clone());
        body["stream"] = Value::Bool(false);
        if let Some(object) = body.as_object_mut() {
            object.remove("stream_options");
        }
        runtime::inject(&mut body, &provider.shortname, &candidate.model);

        let builder = match provider.api {
            ApiKind::OpenAi => {
                let url = join_url(&provider.base_url, "/v1/chat/completions");
                let mut builder = self
                    .client
                    .post(&url)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(serde_json::to_vec(&body).expect("JSON values always serialize"));
                if let Some(key) = provider.api_key() {
                    builder = builder.bearer_auth(key);
                }
                builder
            }
            ApiKind::Anthropic => {
                let translated = anthropic::request_body(&body, &candidate.model)
                    .map_err(AttemptError::Client)?;
                let url = join_url(&provider.base_url, "/v1/messages");
                let mut builder = self
                    .client
                    .post(&url)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", anthropic::ANTHROPIC_VERSION)
                    .body(serde_json::to_vec(&translated).expect("JSON values always serialize"));
                if let Some(key) = provider.api_key() {
                    builder = builder.header("x-api-key", key);
                }
                builder
            }
            other => {
                return Err(failure(format!(
                    "`{}` is not a chat provider",
                    other.as_str()
                )));
            }
        };

        let response = match self.send(builder).await {
            Ok(response) => response,
            Err(error) => return Err(failure(error)),
        };
        let status = response.status();
        if should_failover(status) {
            let error = refusal(status, response).await;
            tracing::warn!(
                provider = %provider.shortname,
                model = %candidate.model,
                error = %error,
                "provider refused the request, trying next"
            );
            return Err(failure(error));
        }

        // Headers are not the answer: a provider that sends them and then stalls
        // must not hold the request open past the timeout.
        let bytes = match tokio::time::timeout(attempt_timeout, response.bytes()).await {
            Ok(Ok(bytes)) => bytes,
            Ok(Err(err)) => {
                return Err(failure(format!("reading provider response failed: {err}")));
            }
            Err(_) => {
                return Err(failure(format!(
                    "provider sent the body too late ({}s timeout)",
                    attempt_timeout.as_secs().max(1)
                )));
            }
        };
        if !status.is_success() {
            let detail = error_detail(&String::from_utf8_lossy(&bytes)).unwrap_or_default();
            return Err(failure(format!("HTTP {status}: {detail}")));
        }
        let value: Value = serde_json::from_slice(&bytes)
            .map_err(|err| failure(format!("provider returned invalid JSON: {err}")))?;
        let body = if provider.api == ApiKind::Anthropic {
            anthropic::response_body(&value, &candidate.model)
        } else {
            value
        };
        Ok(Completion {
            provider: provider.shortname.clone(),
            model: candidate.model.clone(),
            body,
        })
    }

    /// Send the request to the first provider that accepts it.
    pub async fn proxy(&self, request: ForwardRequest) -> Result<Upstream, RouterError> {
        // `max_context` is enforced here, before failover, so every attempt sees
        // the same trimmed prompt and no provider receives the router-only field.
        let (parsed, trim) = match serde_json::from_slice::<Value>(&request.body) {
            Ok(mut value) if value.is_object() => {
                let trim = context::apply(&mut value);
                (Some(value), trim)
            }
            _ => (None, None),
        };
        if let Some(trim) = trim.filter(|trim| trim.changed()) {
            tracing::info!(
                dropped_messages = trim.dropped_messages,
                before_tokens = trim.before_tokens,
                after_tokens = trim.after_tokens,
                "trimmed the prompt to the requested context window"
            );
        }
        let body = match &parsed {
            Some(value) => Bytes::from(
                serde_json::to_vec(value).expect("serializing a JSON value cannot fail"),
            ),
            None => request.body.clone(),
        };
        let requested = parsed
            .as_ref()
            .and_then(|body| body.get("model"))
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|model| !model.is_empty())
            .map(str::to_owned);

        let candidates = self.config.candidates(requested.as_deref())?;

        let stream = parsed
            .as_ref()
            .and_then(|body| body.get("stream"))
            .and_then(Value::as_bool)
            .unwrap_or(false);
        let include_usage = parsed
            .as_ref()
            .and_then(|body| body.get("stream_options"))
            .and_then(|options| options.get("include_usage"))
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let mut failures = Vec::with_capacity(candidates.len());
        // The list is walked in order, restarting from the top after every
        // failure; a walk ends once every model has been tried or skipped.
        for walk in 0..WALKS {
            if walk > 0 {
                self.clear_timeouts(&candidates);
            }
            let mut tried = vec![false; candidates.len()];
            while let Some((index, candidate)) = self.next_candidate(&candidates, &tried) {
                tried[index] = true;
                match self
                    .attempt(
                        &request,
                        &body,
                        candidate,
                        parsed.as_ref(),
                        trim,
                        stream,
                        include_usage,
                    )
                    .await
                {
                    Ok(upstream) => return Ok(upstream),
                    Err(AttemptError::Skip(failure)) => {
                        self.fail(candidate);
                        failures.push(failure);
                    }
                    Err(AttemptError::Client(message)) => {
                        return Err(RouterError::BadRequest(message));
                    }
                }
            }
        }
        Err(RouterError::AllFailed(dedupe(failures)))
    }

    #[allow(clippy::too_many_arguments)]
    async fn attempt(
        &self,
        request: &ForwardRequest,
        body: &Bytes,
        candidate: &Candidate<'_>,
        parsed: Option<&Value>,
        trim: Option<context::Trim>,
        stream: bool,
        include_usage: bool,
    ) -> Result<Upstream, AttemptError> {
        let provider = candidate.provider;
        let failure = |error: String| {
            AttemptError::Skip(Failure {
                provider: provider.shortname.clone(),
                model: candidate.model.clone(),
                error,
            })
        };

        match provider.api {
            ApiKind::OpenAi => {
                let url = join_url(&provider.base_url, &request.path_and_query);
                let mut builder = self.client.request(request.method.clone(), &url);
                for (name, value) in &request.headers {
                    if !STRIPPED_HEADERS.contains(&name.as_str()) {
                        builder = builder.header(name, value);
                    }
                }
                if let Some(key) = provider.api_key() {
                    builder = builder.bearer_auth(key);
                }
                let rewritten = rewrite_body(body, parsed, &candidate.model);
                let rewritten = if is_chat_completions(&request.path_and_query) {
                    with_runtime(rewritten, &provider.shortname, &candidate.model)
                } else {
                    rewritten
                };
                let response = match self.send(builder.body(rewritten)).await {
                    Ok(response) => response,
                    Err(error) => return Err(failure(error)),
                };

                let status = response.status();
                if should_failover(status) {
                    let error = refusal(status, response).await;
                    tracing::warn!(
                        provider = %provider.shortname,
                        model = %candidate.model,
                        error = %error,
                        "provider refused the request, trying next"
                    );
                    return Err(failure(error));
                }
                Ok(Upstream {
                    provider: provider.shortname.clone(),
                    model: candidate.model.clone(),
                    status,
                    content_type: content_type_of(&response),
                    body: Body::from_stream(response.bytes_stream()),
                    context: trim,
                })
            }
            ApiKind::Anthropic => {
                let path = request
                    .path_and_query
                    .split('?')
                    .next()
                    .unwrap_or_default();
                if path.trim_end_matches('/') != "/v1/chat/completions" {
                    return Err(failure(format!(
                        "anthropic providers only support /v1/chat/completions, not `{path}`"
                    )));
                }
                let Some(openai_body) = parsed else {
                    return Err(AttemptError::Client(
                        "request body must be a JSON object".to_owned(),
                    ));
                };
                let mut openai_body = openai_body.clone();
                runtime::inject(&mut openai_body, &provider.shortname, &candidate.model);
                let translated = anthropic::request_body(&openai_body, &candidate.model)
                    .map_err(AttemptError::Client)?;

                let url = join_url(&provider.base_url, "/v1/messages");
                let mut builder = self
                    .client
                    .post(&url)
                    .header(header::CONTENT_TYPE, "application/json")
                    .header("anthropic-version", anthropic::ANTHROPIC_VERSION)
                    .body(serde_json::to_vec(&translated).expect("JSON values always serialize"));
                if let Some(beta) = request.headers.get("anthropic-beta") {
                    builder = builder.header("anthropic-beta", beta);
                }
                if let Some(key) = provider.api_key() {
                    builder = builder.header("x-api-key", key);
                }
                let response = match self.send(builder).await {
                    Ok(response) => response,
                    Err(error) => return Err(failure(error)),
                };

                let status = response.status();
                if should_failover(status) {
                    let error = refusal(status, response).await;
                    tracing::warn!(
                        provider = %provider.shortname,
                        model = %candidate.model,
                        error = %error,
                        "provider refused the request, trying next"
                    );
                    return Err(failure(error));
                }

                if stream && status.is_success() {
                    return Ok(Upstream {
                        provider: provider.shortname.clone(),
                        model: candidate.model.clone(),
                        status,
                        content_type: "text/event-stream".to_owned(),
                        context: trim,
                        body: Body::from_stream(anthropic::stream(
                            response,
                            candidate.model.clone(),
                            include_usage,
                        )),
                    });
                }

                let bytes = match tokio::time::timeout(
                    self.options.attempt_timeout,
                    response.bytes(),
                )
                .await
                {
                    Ok(Ok(bytes)) => bytes,
                    Ok(Err(err)) => {
                        return Err(failure(format!("reading provider response failed: {err}")));
                    }
                    Err(_) => {
                        return Err(failure(format!(
                            "provider sent the body too late ({}s timeout)",
                            self.options.attempt_timeout.as_secs().max(1)
                        )));
                    }
                };
                let message: Value = serde_json::from_slice(&bytes)
                    .map_err(|err| failure(format!("provider returned invalid JSON: {err}")))?;
                let translated = anthropic::response_body(&message, &candidate.model);
                Ok(Upstream {
                    provider: provider.shortname.clone(),
                    model: candidate.model.clone(),
                    status,
                    content_type: "application/json".to_owned(),
                    body: Body::from(Bytes::from(translated.to_string())),
                    context: trim,
                })
            }
            other => Err(failure(format!(
                "`{}` is not a chat provider",
                other.as_str()
            ))),
        }
    }

    async fn send(&self, builder: reqwest::RequestBuilder) -> Result<reqwest::Response, String> {
        let limit = self.options.attempt_timeout;
        match tokio::time::timeout(limit, builder.send()).await {
            Ok(Ok(response)) => Ok(response),
            Ok(Err(err)) => Err(if err.is_connect() {
                format!("connection failed: {err}")
            } else if err.is_timeout() {
                format!("request timed out: {err}")
            } else {
                err.to_string()
            }),
            Err(_) => Err(format!("no response after {}s", limit.as_secs().max(1))),
        }
    }
}

/// `HTTP 404 Not Found: model ... does not exist` — the status plus whatever
/// the provider said, because a bare status never tells you which model name or
/// base URL it disliked.
async fn refusal(status: StatusCode, response: reqwest::Response) -> String {
    let raw = match tokio::time::timeout(Duration::from_secs(2), response.text()).await {
        Ok(Ok(text)) => text,
        _ => String::new(),
    };
    match error_detail(&raw) {
        Some(detail) => format!("HTTP {status}: {detail}"),
        None => format!("HTTP {status}"),
    }
}

/// Best human-readable message in an error body (JSON, plain text, HTML page).
fn error_detail(raw: &str) -> Option<String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return None;
    }
    // A JSON body without a recognisable error message says nothing useful, so
    // it is dropped rather than dumped into the failure list.
    let detail = match serde_json::from_str::<Value>(raw) {
        Ok(value) => json_error_message(&value)?,
        Err(_) => raw.to_owned(),
    };
    let collapsed = detail.split_whitespace().collect::<Vec<_>>().join(" ");
    Some(truncate(&collapsed, 300))
}

/// `{"error":{"message":"…","code":"…"}}`, `{"error":"…"}` and
/// `{"detail":"…"}` are the shapes OpenAI, Anthropic and gateways use.
fn json_error_message(value: &Value) -> Option<String> {
    let error = value.get("error").unwrap_or(value);
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .or_else(|| error.get("detail").and_then(Value::as_str))
        .or_else(|| error.as_str())?;
    match error.get("code").and_then(Value::as_str) {
        Some(code) if !code.is_empty() => Some(format!("{message} ({code})")),
        _ => Some(message.to_owned()),
    }
}

fn truncate(text: &str, max: usize) -> String {
    match text.char_indices().nth(max) {
        Some((end, _)) => format!("{}…", &text[..end]),
        None => text.to_owned(),
    }
}

fn content_type_of(response: &reqwest::Response) -> String {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .unwrap_or("application/json")
        .to_owned()
}

/// Whether another provider is worth trying for this status.
///
/// Only `200 OK` and `304 Not Modified` are treated as an answer. Every other
/// status — including `400`, which is what one provider says when it dislikes a
/// prompt, a tool schema or a model name that the next provider accepts — means
/// this attempt did not serve the request, so failover continues.
fn should_failover(status: StatusCode) -> bool {
    status != StatusCode::OK && status != StatusCode::NOT_MODIFIED
}

/// Is this path a chat completion, where a system message makes sense?
fn is_chat_completions(path_and_query: &str) -> bool {
    let path = path_and_query.split('?').next().unwrap_or_default();
    path.trim_end_matches('/') == "/v1/chat/completions"
}

/// Add the date/model runtime block to an already rewritten chat body.
fn with_runtime(body: Bytes, provider: &str, model: &str) -> Bytes {
    let Ok(mut value) = serde_json::from_slice::<Value>(&body) else {
        return body;
    };
    if !runtime::inject(&mut value, provider, model) {
        return body;
    }
    Bytes::from(serde_json::to_vec(&value).expect("serializing a JSON value cannot fail"))
}

/// Replace the `model` field with the name this provider expects.
fn rewrite_body(body: &Bytes, parsed: Option<&Value>, model: &str) -> Bytes {
    let Some(parsed) = parsed else {
        return body.clone();
    };
    if parsed.get("model").and_then(Value::as_str) == Some(model) {
        return body.clone();
    }
    let mut parsed = parsed.clone();
    parsed["model"] = Value::String(model.to_owned());
    // A `Value` built from parsed JSON always serializes.
    Bytes::from(serde_json::to_vec(&parsed).expect("serializing a JSON value cannot fail"))
}

/// Join a provider base URL with a request path, stripping a duplicated
/// version segment (`https://api.openai.com/v1` + `/v1/chat/completions`).
fn join_url(base_url: &str, path_and_query: &str) -> String {
    let base = base_url.trim_end_matches('/');
    let Some(path) = path_and_query.strip_prefix('/') else {
        return format!("{base}/{path_and_query}");
    };
    let (first, rest) = match path.split_once('/') {
        Some((first, rest)) => (first, Some(rest)),
        None => (path, None),
    };
    if !first.is_empty() && base.ends_with(&format!("/{first}")) {
        return match rest {
            Some(rest) => format!("{base}/{rest}"),
            None => base.to_owned(),
        };
    }
    format!("{base}/{path}")
}

fn summarize(failures: &[Failure]) -> String {
    failures
        .iter()
        .map(|f| format!("{}/{}: {}", f.provider, f.model, f.error))
        .collect::<Vec<_>>()
        .join("; ")
}

/// One entry per model, keeping the most recent error: three walks would
/// otherwise report the same model three times.
fn dedupe(failures: Vec<Failure>) -> Vec<Failure> {
    let mut unique: Vec<Failure> = Vec::with_capacity(failures.len());
    for failure in failures {
        match unique
            .iter_mut()
            .find(|seen| seen.provider == failure.provider && seen.model == failure.model)
        {
            Some(seen) => *seen = failure,
            None => unique.push(failure),
        }
    }
    unique
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use axum::Router as HttpRouter;
    use axum::extract::State;
    use axum::response::{IntoResponse, Response};
    use axum::routing::any;
    use serde_json::json;

    use super::*;

    #[derive(Clone)]
    struct MockState {
        status: StatusCode,
        body: &'static str,
        seen: Arc<Mutex<Vec<String>>>,
    }

    async fn mock_handler(State(state): State<MockState>, body: String) -> Response {
        state.seen.lock().unwrap().push(body);
        (state.status, state.body).into_response()
    }

    /// Start a fake provider. Returns its base URL and the bodies it received.
    async fn spawn_provider(status: StatusCode) -> (String, Arc<Mutex<Vec<String>>>) {
        spawn_with_body(status, r#"{"ok":true}"#).await
    }

    async fn spawn_with_body(
        status: StatusCode,
        body: &'static str,
    ) -> (String, Arc<Mutex<Vec<String>>>) {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let app = HttpRouter::new()
            .route("/v1/chat/completions", any(mock_handler))
            .with_state(MockState {
                status,
                body,
                seen: seen.clone(),
            });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{addr}/v1"), seen)
    }

    fn chat_request(model: &str) -> ForwardRequest {
        ForwardRequest {
            method: Method::POST,
            path_and_query: "/v1/chat/completions".to_owned(),
            headers: HeaderMap::new(),
            body: Bytes::from(
                json!({"model": model, "messages": [{"role": "user", "content": "hi"}]})
                    .to_string(),
            ),
        }
    }

    fn config(providers: &[(&str, &str)]) -> Config {
        let mut text = String::from("[providers]\n");
        for (shortname, base_url) in providers {
            text.push_str(&format!("openai,{base_url},{shortname}\n"));
        }
        text.push_str("[models]\n");
        for (shortname, _) in providers {
            text.push_str(&format!("{shortname}/gpt-test\n"));
        }
        Config::parse(&text).unwrap()
    }

    fn router(config: Config) -> Router {
        Router::new(
            config,
            Options {
                attempt_timeout: Duration::from_secs(10),
                model_timeout: Duration::from_secs(30),
            },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn fails_over_past_a_rate_limited_provider() {
        let (first, first_seen) = spawn_provider(StatusCode::TOO_MANY_REQUESTS).await;
        let (second, second_seen) = spawn_provider(StatusCode::OK).await;
        let router = router(config(&[("p1", &first), ("p2", &second)]));

        let upstream = router.proxy(chat_request("p1/gpt-test")).await.unwrap();

        assert_eq!(upstream.provider, "p2");
        assert_eq!(upstream.model, "gpt-test");
        assert_eq!(upstream.status, StatusCode::OK);
        assert_eq!(first_seen.lock().unwrap().len(), 1);
        assert_eq!(second_seen.lock().unwrap().len(), 1);
        // The provider prefix is stripped before forwarding.
        let forwarded: Value = serde_json::from_str(&second_seen.lock().unwrap()[0]).unwrap();
        assert_eq!(forwarded["model"], "gpt-test");
    }

    /// Free tiers limit models, not providers: a throttled model must not hide
    /// its neighbours on the same provider.
    #[tokio::test]
    async fn parks_only_the_rate_limited_model() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let app = HttpRouter::new().route(
            "/v1/chat/completions",
            any(move |body: String| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body.clone());
                    let request: Value = serde_json::from_str(&body).unwrap();
                    let status = if request["model"] == "a" {
                        StatusCode::TOO_MANY_REQUESTS
                    } else {
                        StatusCode::OK
                    };
                    (status, r#"{"ok":true}"#).into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let config = Config::parse(&format!(
            "[providers]\nopenai,http://{addr}/v1,p1\n[models]\np1/a\np1/b\n"
        ))
        .unwrap();
        let router = router(config);

        // `a` is limited; the same request still gets an answer from `b`.
        let first = router.proxy(chat_request("p1/a")).await.unwrap();
        assert_eq!(first.model, "b");

        // The second request skips the parked model `a` and reuses `b` on the
        // very same provider.
        let second = router.proxy(chat_request("p1/a")).await.unwrap();
        assert_eq!(second.model, "b");

        let models: Vec<String> = seen
            .lock()
            .unwrap()
            .iter()
            .map(|body| serde_json::from_str::<Value>(body).unwrap()["model"].to_string())
            .collect();
        assert_eq!(
            models.iter().filter(|model| *model == "\"a\"").count(),
            1,
            "the limited model was contacted again: {models:?}"
        );
    }

    /// The scenario from the brief: once one request has learned p1 is rate
    /// limited, concurrent requests go straight to p2 without queueing.
    #[tokio::test]
    async fn concurrent_requests_skip_a_parked_provider() {
        let (first, first_seen) = spawn_provider(StatusCode::TOO_MANY_REQUESTS).await;
        let (second, second_seen) = spawn_provider(StatusCode::OK).await;
        let router = Arc::new(router(config(&[("p1", &first), ("p2", &second)])));

        // Prime the rate-limit knowledge.
        router.proxy(chat_request("auto")).await.unwrap();
        assert_eq!(first_seen.lock().unwrap().len(), 1);

        let (a, b) = tokio::join!(
            router.proxy(chat_request("auto")),
            router.proxy(chat_request("auto")),
        );
        assert_eq!(a.unwrap().provider, "p2");
        assert_eq!(b.unwrap().provider, "p2");
        // p1 was never contacted again while parked.
        assert_eq!(first_seen.lock().unwrap().len(), 1);
        assert_eq!(second_seen.lock().unwrap().len(), 3);
    }

    /// A failed model is skipped by later requests until its timeout expires.
    #[tokio::test]
    async fn skips_a_failed_model_until_its_timeout_expires() {
        let (first, first_seen) = spawn_provider(StatusCode::INTERNAL_SERVER_ERROR).await;
        let (second, _) = spawn_provider(StatusCode::OK).await;
        let router = router(config(&[("p1", &first), ("p2", &second)]));

        router.proxy(chat_request("auto")).await.unwrap();
        router.proxy(chat_request("auto")).await.unwrap();
        assert_eq!(first_seen.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn fails_over_past_an_unreachable_provider() {
        let (second, _) = spawn_provider(StatusCode::OK).await;
        let router = router(config(&[("dead", "http://127.0.0.1:1/v1"), ("p2", &second)]));

        let upstream = router.proxy(chat_request("auto")).await.unwrap();
        assert_eq!(upstream.provider, "p2");
    }

    #[tokio::test]
    async fn fails_over_on_client_errors() {
        // A provider may reject a request (a 400) that the next provider serves:
        // the router must keep going instead of handing the 400 to the client.
        for status in [
            StatusCode::BAD_REQUEST,
            StatusCode::UNPROCESSABLE_ENTITY,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::UNAUTHORIZED,
        ] {
            let (first, _) = spawn_provider(status).await;
            let (second, second_seen) = spawn_provider(StatusCode::OK).await;
            let router = router(config(&[("p1", &first), ("p2", &second)]));

            let upstream = router.proxy(chat_request("auto")).await.unwrap();
            assert_eq!(upstream.provider, "p2", "{status} was not failed over");
            assert_eq!(upstream.status, StatusCode::OK);
            assert!(!second_seen.lock().unwrap().is_empty());
        }
    }

    #[tokio::test]
    async fn reports_every_attempt_when_all_providers_reject_the_request() {
        let (first, _) = spawn_provider(StatusCode::BAD_REQUEST).await;
        let (second, _) = spawn_provider(StatusCode::BAD_REQUEST).await;
        let router = router(config(&[("p1", &first), ("p2", &second)]));

        let err = router.proxy(chat_request("auto")).await.unwrap_err();
        let RouterError::AllFailed(failures) = err else {
            panic!("expected every attempt to be reported: {err}");
        };
        assert_eq!(failures.len(), 2);
        assert!(failures.iter().all(|f| f.error.contains("400")));
    }

    /// Every model is tried once per walk, and the list is walked three times
    /// before the request fails.
    #[tokio::test]
    async fn walks_the_whole_list_in_order_three_times() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let sink = seen.clone();
        let app = HttpRouter::new().route(
            "/v1/chat/completions",
            any(move |body: String| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(body);
                    (StatusCode::TOO_MANY_REQUESTS, r#"{"ok":true}"#).into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let config = Config::parse(&format!(
            "[providers]\nopenai,http://{addr}/v1,p1\n[models]\np1/a\np1/b\n"
        ))
        .unwrap();
        let router = Router::new(
            config,
            Options {
                attempt_timeout: Duration::from_secs(5),
                model_timeout: Duration::from_millis(20),
            },
        )
        .unwrap();

        let err = router.proxy(chat_request("auto")).await.unwrap_err();
        // Three walks, one entry per model: duplicates across walks collapse.
        let RouterError::AllFailed(failures) = err else {
            panic!("expected an aggregated failure");
        };
        assert_eq!(failures.len(), 2, "{failures:?}");

        let bodies = seen.lock().unwrap();
        assert_eq!(bodies.len(), 2 * WALKS);
        let models: Vec<String> = bodies
            .iter()
            .map(|body| serde_json::from_str::<Value>(body).unwrap()["model"].to_string())
            .collect();
        // One attempt per model in each of the three walks, in config order.
        for offset in 0..WALKS {
            assert_eq!(models[offset * 2], "\"a\"", "{models:?}");
            assert_eq!(models[offset * 2 + 1], "\"b\"", "{models:?}");
        }
    }

    #[tokio::test]
    async fn reports_every_failed_attempt() {
        let (first, _) = spawn_provider(StatusCode::TOO_MANY_REQUESTS).await;
        let (second, _) = spawn_provider(StatusCode::INTERNAL_SERVER_ERROR).await;
        let router = router(config(&[("p1", &first), ("p2", &second)]));

        let err = router.proxy(chat_request("auto")).await.unwrap_err();
        let message = err.to_string();
        assert!(message.contains("p1/gpt-test: HTTP 429"), "{message}");
        assert!(message.contains("p2/gpt-test: HTTP 500"), "{message}");
    }

    /// A community endpoint that never answers is skipped like any other
    /// failure: the next request goes straight to the next model instead of
    /// waiting out the same stall again.
    #[tokio::test]
    async fn parks_a_model_that_never_answers() {
        let slow_seen = Arc::new(Mutex::new(0usize));
        let slow = slow_seen.clone();
        let app = HttpRouter::new().route(
            "/v1/chat/completions",
            any(move || {
                let slow = slow.clone();
                async move {
                    *slow.lock().unwrap() += 1;
                    tokio::time::sleep(Duration::from_secs(10)).await;
                    (StatusCode::OK, r#"{"ok":true}"#).into_response()
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let (second, _) = spawn_provider(StatusCode::OK).await;
        let router = Router::new(
            config(&[("slow", &format!("http://{addr}/v1")), ("p2", &second)]),
            Options {
                attempt_timeout: Duration::from_secs(1),
                model_timeout: Duration::from_secs(30),
            },
        )
        .unwrap();

        router.proxy(chat_request("auto")).await.unwrap();
        assert_eq!(*slow_seen.lock().unwrap(), 1);

        // The stalled model is parked, so the next request reaches p2 directly.
        router.proxy(chat_request("auto")).await.unwrap();
        assert_eq!(*slow_seen.lock().unwrap(), 1, "the stalled model was retried");
    }

    #[tokio::test]
    async fn reports_why_the_provider_refused() {
        let (first, _) = spawn_with_body(
            StatusCode::NOT_FOUND,
            r#"{"error":{"message":"model z-ai/glm-5.3-flash-free not found","code":"model_not_found"}}"#,
        )
        .await;
        let router = router(config(&[("p1", &first)]));

        let err = router.proxy(chat_request("auto")).await.unwrap_err();
        let message = err.to_string();
        assert!(message.contains("HTTP 404"), "{message}");
        assert!(message.contains("not found"), "{message}");
        assert!(message.contains("model_not_found"), "{message}");
    }

    /// Exhausting the list clears every timeout and walks it again, twice.
    #[tokio::test]
    async fn clears_timeouts_and_repeats_the_walk_twice() {
        let (first, first_seen) = spawn_provider(StatusCode::TOO_MANY_REQUESTS).await;
        // The timeout outlasts the request, so only the end-of-list reset can
        // make the model reachable again.
        let router = Router::new(
            config(&[("p1", &first)]),
            Options {
                attempt_timeout: Duration::from_secs(10),
                model_timeout: Duration::from_secs(60),
            },
        )
        .unwrap();

        assert!(router.proxy(chat_request("auto")).await.is_err());
        assert_eq!(first_seen.lock().unwrap().len(), WALKS);

        // The second request starts with the model still parked, so its first
        // walk contacts nobody; the two remaining walks retry it each.
        assert!(router.proxy(chat_request("auto")).await.is_err());
        assert_eq!(first_seen.lock().unwrap().len(), WALKS + (WALKS - 1));
    }

    #[test]
    fn joins_base_urls_without_duplicating_the_version_segment() {
        assert_eq!(
            join_url("https://api.openai.com/v1", "/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_url("https://api.openai.com", "/v1/chat/completions"),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            join_url("https://proxy.dev/openai/v1/", "/v1/chat/completions?stream=true"),
            "https://proxy.dev/openai/v1/chat/completions?stream=true"
        );
        assert_eq!(
            join_url("https://api.anthropic.com", "/v1/messages"),
            "https://api.anthropic.com/v1/messages"
        );
    }

    #[test]
    fn extracts_error_messages_from_every_error_shape() {
        assert_eq!(
            error_detail(r#"{"error":{"message":"model x not found","code":"model_not_found"}}"#),
            Some("model x not found (model_not_found)".to_owned())
        );
        assert_eq!(
            error_detail(r#"{"error":"insufficient credit"}"#),
            Some("insufficient credit".to_owned())
        );
        assert_eq!(
            error_detail(r#"{"detail":"Not Found"}"#),
            Some("Not Found".to_owned())
        );
        assert_eq!(
            error_detail("<html> 404   page not found </html>"),
            Some("<html> 404 page not found </html>".to_owned())
        );
        // A JSON body with no message should not be dumped verbatim.
        assert_eq!(error_detail(r#"{"ok":true}"#), None);
        assert_eq!(error_detail("   "), None);
    }

    #[tokio::test]
    async fn trims_to_max_context_and_strips_the_field() {
        let (provider, seen) = spawn_provider(StatusCode::OK).await;
        let router = router(config(&[("p1", &provider)]));
        let messages: Vec<Value> = (0..30)
            .map(|index| {
                json!({
                    "role": if index % 2 == 0 { "user" } else { "assistant" },
                    "content": "x".repeat(400)
                })
            })
            .collect();
        let request = ForwardRequest {
            method: Method::POST,
            path_and_query: "/v1/chat/completions".to_owned(),
            headers: HeaderMap::new(),
            body: Bytes::from(
                json!({
                    "model": "p1/gpt-test",
                    "max_context": 1500,
                    "messages": messages
                })
                .to_string(),
            ),
        };

        let upstream = router.proxy(request).await.unwrap();
        let forwarded: Value = serde_json::from_str(&seen.lock().unwrap()[0]).unwrap();
        assert!(forwarded.get("max_context").is_none());
        let messages = forwarded["messages"].as_array().unwrap();
        // The router prepends its own runtime block after trimming.
        assert_eq!(messages[0]["role"], "system");
        assert!(messages[0]["content"].as_str().unwrap().contains("[runtime]"));
        let kept = messages.len() - 1;
        assert!(kept < 30, "nothing was dropped: {forwarded}");
        let trim = upstream.context.unwrap();
        assert_eq!(trim.dropped_messages, 30 - kept);
        let response = IntoResponse::into_response(upstream);
        assert_eq!(
            response.headers()["x-router-context-dropped"],
            trim.dropped_tokens.to_string()
        );
        assert_eq!(
            response.headers()["x-router-context-messages"],
            trim.dropped_messages.to_string()
        );
    }

    #[test]
    fn rewrites_the_model_field_only() {
        let body = Bytes::from(r#"{"model":"p1/gpt","temperature":0.2}"#);
        let parsed: Value = serde_json::from_slice(&body).unwrap();
        let rewritten = rewrite_body(&body, Some(&parsed), "gpt");
        let value: Value = serde_json::from_slice(&rewritten).unwrap();
        assert_eq!(value["model"], "gpt");
        assert_eq!(value["temperature"], 0.2);
    }

    #[test]
    fn passes_non_json_bodies_through() {
        let body = Bytes::from_static(b"not json");
        assert_eq!(rewrite_body(&body, None, "gpt"), body);
    }
}

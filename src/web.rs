//! Keyless web search and page reading, behind the router's own endpoints.
//!
//! Search providers are tried in config order and never wait on each other:
//! a provider that answers with a rate limit, an error or a page without
//! results is abandoned and the next one is tried. The shipped dialects are
//! keyless:
//!
//! * `duckduckgo` — DuckDuckGo's no-JS HTML results page. No key, no quota;
//!   it is the general-purpose option and occasionally shows a bot check,
//!   which is why it fails over to the next provider instead of erroring.
//! * `wikipedia` — MediaWiki's search API. No key, stable and generous, and a
//!   good fallback for factual lookups.
//! * `jina-search` — `s.jina.ai` (needs a free key; text results).
//!
//! `jina` (`r.jina.ai`) is the fetch provider: it renders any URL as clean
//! text, keyless at 20 requests/minute, or 500/minute with a free key.


use axum::http::{StatusCode, header};
use serde_json::Value;
use thiserror::Error;

use crate::config::{ApiKind, Provider};
use crate::router::{now_ms, Router};

/// Longest query forwarded to a search provider.
const MAX_QUERY_CHARS: usize = 400;
/// Fetched pages are fed to models, so cap them at roughly 6k tokens.
const MAX_FETCH_CHARS: usize = 24_000;
const DEFAULT_RESULTS: usize = 6;
const MAX_RESULTS: usize = 10;
const SNIPPET_CHARS: usize = 400;
/// A page shorter than this is almost certainly an error page.
const MIN_FETCH_CHARS: usize = 80;

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 \
                          (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

#[derive(Debug, Error)]
pub enum WebError {
    #[error("no {0} providers are configured (add one to [providers], e.g. `duckduckgo,https://html.duckduckgo.com,ddg`)")]
    NoProviders(&'static str),
    #[error("the query is empty")]
    EmptyQuery,
    #[error("`{0}` is not an http(s) URL")]
    BadUrl(String),
    #[error("`{0}` points at a local or private address, which web_fetch refuses")]
    PrivateAddress(String),
    #[error("every provider failed ({})", .0.join("; "))]
    AllFailed(Vec<String>),
}

#[derive(Debug, Clone)]
pub struct SearchHit {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

#[derive(Debug)]
pub struct SearchOutcome {
    pub provider: String,
    pub results: Vec<SearchHit>,
    /// The results as one block of text, ready to hand to a model.
    pub text: String,
}

#[derive(Debug)]
pub struct FetchOutcome {
    pub provider: String,
    pub url: String,
    pub title: String,
    pub text: String,
    pub truncated: bool,
}

/// Why one provider did not answer.
enum Skip {
    /// Rate limited: park the provider so siblings skip it, then fail over.
    RateLimited(String),
    Other(String),
}

impl Router {
    /// Search the web with the first configured search provider that answers.
    pub async fn web_search(&self, query: &str, limit: usize) -> Result<SearchOutcome, WebError> {
        let query = query.trim();
        if query.is_empty() {
            return Err(WebError::EmptyQuery);
        }
        let query: String = query.chars().take(MAX_QUERY_CHARS).collect();
        let limit = if limit == 0 {
            DEFAULT_RESULTS
        } else {
            limit.min(MAX_RESULTS)
        };

        let providers: Vec<&Provider> = self
            .config()
            .search_providers()
            .map(|(_, provider)| provider)
            .collect();
        if providers.is_empty() {
            return Err(WebError::NoProviders("web search"));
        }

        let mut failures = Vec::new();
        let mut attempted = 0;
        for provider in &providers {
            if !self.web_slot(provider).is_available(now_ms()) {
                continue;
            }
            attempted += 1;
            match self.search_one(provider, &query, limit).await {
                Ok(outcome) => {
                    tracing::info!(
                        provider = %provider.shortname,
                        results = outcome.results.len(),
                        query = %query,
                        "web search answered"
                    );
                    return Ok(outcome);
                }
                Err(Skip::RateLimited(error)) => {
                    self.park(provider);
                    failures.push(format!("{}: {error}", provider.shortname));
                }
                Err(Skip::Other(error)) => {
                    failures.push(format!("{}: {error}", provider.shortname));
                }
            }
        }
        // Everything was parked by a sibling request: try the first anyway.
        if attempted == 0
            && let Some(provider) = providers.first()
        {
            match self.search_one(provider, &query, limit).await {
                Ok(outcome) => return Ok(outcome),
                Err(Skip::RateLimited(error)) | Err(Skip::Other(error)) => {
                    failures.push(format!("{}: {error}", provider.shortname));
                }
            }
        }
        Err(WebError::AllFailed(failures))
    }

    /// Read a URL as clean text with the first configured fetch provider.
    pub async fn web_fetch(&self, url: &str) -> Result<FetchOutcome, WebError> {
        let url = url.trim();
        reject_local(url)?;

        let providers: Vec<&Provider> = self
            .config()
            .fetch_providers()
            .map(|(_, provider)| provider)
            .collect();
        if providers.is_empty() {
            return Err(WebError::NoProviders("page fetch"));
        }

        let mut failures = Vec::new();
        let mut attempted = 0;
        for provider in &providers {
            if !self.web_slot(provider).is_available(now_ms()) {
                continue;
            }
            attempted += 1;
            match self.fetch_one(provider, url).await {
                Ok(outcome) => {
                    tracing::info!(
                        provider = %provider.shortname,
                        chars = outcome.text.len(),
                        %url,
                        "web fetch answered"
                    );
                    return Ok(outcome);
                }
                Err(Skip::RateLimited(error)) => {
                    self.park(provider);
                    failures.push(format!("{}: {error}", provider.shortname));
                }
                Err(Skip::Other(error)) => {
                    failures.push(format!("{}: {error}", provider.shortname));
                }
            }
        }
        if attempted == 0
            && let Some(provider) = providers.first()
        {
            match self.fetch_one(provider, url).await {
                Ok(outcome) => return Ok(outcome),
                Err(Skip::RateLimited(error)) | Err(Skip::Other(error)) => {
                    failures.push(format!("{}: {error}", provider.shortname));
                }
            }
        }
        Err(WebError::AllFailed(failures))
    }

    fn park(&self, provider: &Provider) {
        self.web_slot(provider).block(self.options.model_timeout);
    }

    async fn search_one(
        &self,
        provider: &Provider,
        query: &str,
        limit: usize,
    ) -> Result<SearchOutcome, Skip> {
        let (hits, text) = match provider.api {
            ApiKind::DuckDuckGo => {
                let url = format!(
                    "{}/html/?q={}&kl=wt-wt",
                    provider.base_url.trim_end_matches('/'),
                    encode(query)
                );
                let body = self.get_text(provider, &url, "text/html").await?;
                let hits = parse_duckduckgo(&body, limit);
                if hits.is_empty() {
                    return Err(Skip::Other(format!(
                        "no results were parsed ({} bytes, the engine may be showing a bot check)",
                        body.len()
                    )));
                }
                let text = format_hits(query, &hits);
                (hits, text)
            }
            ApiKind::Wikipedia => {
                let url = format!(
                    "{}/w/api.php?action=query&list=search&srsearch={}&format=json&utf8=1&srlimit={limit}",
                    provider.base_url.trim_end_matches('/'),
                    encode(query)
                );
                let body = self
                    .get_text(provider, &url, "application/json")
                    .await?;
                let hits = parse_wikipedia(&body, &provider.base_url, limit)?;
                if hits.is_empty() {
                    return Err(Skip::Other("no results".to_owned()));
                }
                let text = format_hits(query, &hits);
                (hits, text)
            }
            ApiKind::JinaSearch => {
                let url = format!(
                    "{}/?q={}",
                    provider.base_url.trim_end_matches('/'),
                    encode(query)
                );
                let body = self.get_text(provider, &url, "text/plain").await?;
                if body.trim().len() < MIN_FETCH_CHARS {
                    return Err(Skip::Other("response was too short to be results".to_owned()));
                }
                let text: String = body.chars().take(MAX_FETCH_CHARS).collect();
                (Vec::new(), text)
            }
            other => {
                return Err(Skip::Other(format!(
                    "`{}` is not a search provider",
                    other.as_str()
                )));
            }
        };
        Ok(SearchOutcome {
            provider: provider.shortname.clone(),
            results: hits,
            text,
        })
    }

    async fn fetch_one(
        &self,
        provider: &Provider,
        url: &str,
    ) -> Result<FetchOutcome, Skip> {
        match provider.api {
            ApiKind::JinaReader => {
                let target = format!("{}/{}", provider.base_url.trim_end_matches('/'), url);
                let body = self.get_text(provider, &target, "text/plain").await?;
                let body = body.trim();
                if body.len() < MIN_FETCH_CHARS {
                    return Err(Skip::Other(format!(
                        "the page came back with only {} characters",
                        body.len()
                    )));
                }
                let truncated = body.chars().count() > MAX_FETCH_CHARS;
                let text: String = body.chars().take(MAX_FETCH_CHARS).collect();
                Ok(FetchOutcome {
                    provider: provider.shortname.clone(),
                    url: url.to_owned(),
                    title: jina_title(&text).unwrap_or_else(|| url.to_owned()),
                    text,
                    truncated,
                })
            }
            other => Err(Skip::Other(format!(
                "`{}` is not a fetch provider",
                other.as_str()
            ))),
        }
    }

    async fn get_text(
        &self,
        provider: &Provider,
        url: &str,
        accept: &str,
    ) -> Result<String, Skip> {
        let mut builder = self
            .client
            .get(url)
            .header(header::ACCEPT, accept)
            .header(header::USER_AGENT, USER_AGENT)
            .header(header::ACCEPT_LANGUAGE, "en-US,en;q=0.9");
        if let Some(key) = provider.api_key() {
            builder = builder.bearer_auth(key);
        }
        let response = match tokio::time::timeout(self.options.attempt_timeout, builder.send()).await
        {
            Ok(Ok(response)) => response,
            Ok(Err(err)) => return Err(Skip::Other(format!("request failed: {err}"))),
            Err(_) => {
                return Err(Skip::Other(format!(
                    "no response after {}s",
                    self.options.attempt_timeout.as_secs()
                )));
            }
        };

        let status = response.status();
        match status {
            StatusCode::TOO_MANY_REQUESTS => {
                self.park(provider);
                return Err(Skip::RateLimited("rate limited".to_owned()));
            }
            s if !s.is_success() => {
                return Err(Skip::Other(format!("HTTP {s}")));
            }
            _ => {}
        }

        response
            .text()
            .await
            .map_err(|err| Skip::Other(format!("reading the response failed: {err}")))
    }
}

/// DuckDuckGo's HTML page, without a DOM library: each result is an anchor with
/// `class="result__a"` (whose `href` carries the real URL in `uddg=`) followed by
/// a `result__snippet` element.
fn parse_duckduckgo(html: &str, limit: usize) -> Vec<SearchHit> {
    let mut hits = Vec::new();
    let mut rest = html;

    while hits.len() < limit {
        let Some(class_at) = rest.find("result__a") else {
            break;
        };
        let tag_start = rest[..class_at].rfind('<').unwrap_or(class_at);
        let Some(open_end) = rest[class_at..].find('>') else {
            break;
        };
        let open_end = class_at + open_end;
        let tag = &rest[tag_start..open_end];
        let href = attribute(tag, "href").map(|raw| unpack_duckduckgo_url(&raw));
        let Some(close) = rest[open_end..].find("</a>") else {
            break;
        };
        let title = clean(&rest[open_end + 1..open_end + close]);
        rest = &rest[open_end + close + 4..];

        // The snippet belongs to this result, so stop at the next one.
        let block_end = rest.find("result__a").unwrap_or(rest.len());
        let snippet = parse_duckduckgo_snippet(&rest[..block_end]);

        if let (Some(url), false) = (href, title.is_empty()) {
            hits.push(SearchHit {
                title,
                url,
                snippet,
            });
        }
    }
    hits
}

fn parse_duckduckgo_snippet(block: &str) -> String {
    let Some(class_at) = block.find("result__snippet") else {
        return String::new();
    };
    let Some(open_end) = block[class_at..].find('>') else {
        return String::new();
    };
    let open_end = class_at + open_end;
    let tail = &block[open_end + 1..];
    let end = tail
        .find("</a>")
        .or_else(|| tail.find("</div>"))
        .unwrap_or(tail.len());
    shorten(&clean(&tail[..end]), SNIPPET_CHARS)
}

/// Wikipedia's `list=search` JSON: `query.search[]` with an HTML snippet.
fn parse_wikipedia(body: &str, base_url: &str, limit: usize) -> Result<Vec<SearchHit>, Skip> {
    let value: Value = serde_json::from_str(body)
        .map_err(|err| Skip::Other(format!("response was not JSON: {err}")))?;
    let base = base_url.trim_end_matches('/');
    Ok(value
        .get("query")
        .and_then(|query| query.get("search"))
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .take(limit)
                .filter_map(|result| {
                    let title = result.get("title")?.as_str()?.to_owned();
                    let slug = title.replace(' ', "_");
                    Some(SearchHit {
                        url: format!("{base}/wiki/{}", encode(&slug)),
                        snippet: shorten(
                            &clean(result.get("snippet").and_then(Value::as_str).unwrap_or_default()),
                            SNIPPET_CHARS,
                        ),
                        title,
                    })
                })
                .collect()
        })
        .unwrap_or_default())
}

fn format_hits(query: &str, hits: &[SearchHit]) -> String {
    let mut text = format!("Web results for \"{query}\":\n");
    for (index, hit) in hits.iter().enumerate() {
        text.push_str(&format!(
            "\n{}. {}\n   {}\n   {}\n",
            index + 1,
            hit.title,
            hit.url,
            hit.snippet
        ));
    }
    text
}

/// `//duckduckgo.com/l/?uddg=<target>&rut=…` → the target URL.
fn unpack_duckduckgo_url(raw: &str) -> String {
    let raw = unescape(raw.trim());
    let target = raw
        .split_once("uddg=")
        .map(|(_, rest)| rest.split('&').next().unwrap_or(rest))
        .map(decode)
        .unwrap_or_else(|| raw.clone());
    if target.starts_with("//") {
        format!("https:{target}")
    } else {
        target
    }
}

/// Jina's reader output starts with `Title: …`, `URL Source: …` and
/// `Markdown Content:` lines.
fn jina_title(text: &str) -> Option<String> {
    text.lines()
        .next()
        .and_then(|line| line.strip_prefix("Title:"))
        .map(|title| title.trim().to_owned())
        .filter(|title| !title.is_empty())
}

/// Refuse the obvious SSRF shapes: anything that is not http(s), and literal
/// local/private hosts. (The fetch provider performs the request, so it also
/// cannot reach this machine.)
fn reject_local(url: &str) -> Result<(), WebError> {
    let rest = url
        .strip_prefix("https://")
        .or_else(|| url.strip_prefix("http://"))
        .ok_or_else(|| WebError::BadUrl(url.to_owned()))?;
    let authority = rest
        .split(['/', '?', '#'])
        .next()
        .unwrap_or_default()
        .rsplit('@')
        .next()
        .unwrap_or_default();
    let host = if let Some(bracketed) = authority.strip_prefix('[') {
        bracketed.split(']').next().unwrap_or_default()
    } else {
        authority.split(':').next().unwrap_or_default()
    };
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host.is_empty() {
        return Err(WebError::BadUrl(url.to_owned()));
    }
    let local = host == "localhost"
        || host.ends_with(".localhost")
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host == "::1"
        || host == "::"
        || host.starts_with("fc")
        || host.starts_with("fd")
        || host.starts_with("fe80");
    let literal_private = host.parse::<std::net::IpAddr>().is_ok_and(|ip| match ip {
        std::net::IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
        }
        std::net::IpAddr::V6(v6) => v6.is_loopback() || v6.is_unspecified() || v6.is_multicast(),
    });
    if local || literal_private {
        return Err(WebError::PrivateAddress(url.to_owned()));
    }
    Ok(())
}

fn attribute(tag: &str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let start = tag.find(&needle)? + needle.len();
    let end = tag[start..].find('"')? + start;
    Some(tag[start..end].to_owned())
}

fn clean(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut depth = 0usize;
    for ch in html.chars() {
        match ch {
            '<' => depth += 1,
            '>' if depth > 0 => depth -= 1,
            _ if depth == 0 => out.push(ch),
            _ => {}
        }
    }
    unescape(&out)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(pos) = rest.find('&') {
        out.push_str(&rest[..pos]);
        let tail = &rest[pos..];
        let Some(end) = tail.find(';').filter(|end| *end <= 10) else {
            out.push('&');
            rest = &tail[1..];
            continue;
        };
        let entity = &tail[1..end];
        let decoded = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" | "#39" => Some('\''),
            "nbsp" => Some(' '),
            _ => entity
                .strip_prefix("#x")
                .or_else(|| entity.strip_prefix("#X"))
                .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                .or_else(|| entity.strip_prefix('#').and_then(|dec| dec.parse().ok()))
                .and_then(char::from_u32),
        };
        match decoded {
            Some(ch) => {
                out.push(ch);
                rest = &tail[end + 1..];
            }
            None => {
                out.push('&');
                rest = &tail[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut out: String = text.chars().take(max).collect();
    out.push('…');
    out
}

fn encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len() * 3);
    for byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char);
            }
            _ => out.push_str(&format!("%{byte:02X}")),
        }
    }
    out
}

fn decode(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 2 < bytes.len() => match (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                (Some(high), Some(low)) => {
                    out.push(high * 16 + low);
                    i += 3;
                }
                _ => {
                    out.push(bytes[i]);
                    i += 1;
                }
            },
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DDG_HTML: &str = r#"
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%3Fb%3D1&amp;rut=abc">Example <b>Title</b></a>
  <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa">A snippet &amp; more</a>
</div>
<div class="result results_links">
  <a rel="nofollow" class="result__a" href="https://direct.example.com/page">Second &quot;one&quot;</a>
  <div class="result__snippet">Second snippet</div>
</div>
"#;

    #[test]
    fn parses_duckduckgo_results() {
        let hits = parse_duckduckgo(DDG_HTML, 5);
        assert_eq!(hits.len(), 2, "{hits:#?}");
        assert_eq!(hits[0].title, "Example Title");
        assert_eq!(hits[0].url, "https://example.com/a?b=1");
        assert_eq!(hits[0].snippet, "A snippet & more");
        assert_eq!(hits[1].title, "Second \"one\"");
        assert_eq!(hits[1].url, "https://direct.example.com/page");
        assert_eq!(hits[1].snippet, "Second snippet");
    }

    #[test]
    fn duckduckgo_respects_the_limit() {
        assert_eq!(parse_duckduckgo(DDG_HTML, 1).len(), 1);
    }

    #[test]
    fn parses_wikipedia_json() {
        let body = r#"{"query":{"search":[
            {"title":"Rust (programming language)","snippet":"<span class=\"searchmatch\">Rust</span> is &amp; fast"},
            {"title":"Rust","snippet":"other"}
        ]}}"#;
        let hits = parse_wikipedia(body, "https://en.wikipedia.org", 5).ok().unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].url, "https://en.wikipedia.org/wiki/Rust_%28programming_language%29");
        assert_eq!(hits[0].snippet, "Rust is & fast");
    }

    #[test]
    fn local_targets_are_refused() {
        for url in [
            "http://127.0.0.1:8080/x",
            "https://localhost/x",
            "http://10.0.0.5/",
            "http://192.168.1.1/",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
            "http://printer.local/",
            "ftp://example.com/file",
        ] {
            assert!(
                reject_local(url).is_err(),
                "{url} should have been refused"
            );
        }
        for url in ["https://example.com", "http://example.com:8080/a?b=c"] {
            assert!(reject_local(url).is_ok(), "{url} should be allowed");
        }
    }

    #[test]
    fn entities_and_encoding_round_trip() {
        assert_eq!(unescape("a &amp; b &#39;c&#x27; &unknown;"), "a & b 'c' &unknown;");
        assert_eq!(encode("a b&c/d"), "a%20b%26c%2Fd");
        assert_eq!(decode("a%20b%26c%2Fd"), "a b&c/d");
        assert_eq!(decode("%zz"), "%zz");
    }

    #[test]
    fn formats_hits_for_models() {
        let text = format_hits(
            "rust axum",
            &[SearchHit {
                title: "Axum".into(),
                url: "https://docs.rs/axum".into(),
                snippet: "web framework".into(),
            }],
        );
        assert!(text.contains("Web results for \"rust axum\""), "{text}");
        assert!(text.contains("1. Axum"), "{text}");
        assert!(text.contains("https://docs.rs/axum"), "{text}");
    }

    #[test]
    fn jina_titles_are_read_from_the_header() {
        assert_eq!(
            jina_title("Title: Example\n\nURL Source: https://example.com").as_deref(),
            Some("Example")
        );
        assert!(jina_title("no header").is_none());
    }
}

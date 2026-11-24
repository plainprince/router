//! Loading and validation of the router's config file.
//!
//! The format is deliberately tiny and line oriented:
//!
//! ```text
//! # <api type>,<base url>,<shortname>[,<api key | env var name>]
//! [providers]
//! openai,https://api.openai.com/v1,openai
//! openai,https://openrouter.ai/api/v1,orcarouter,sk-orca-...
//! openai,https://api.x.ai/v1,xai,env:XAI_KEY
//!
//! # <provider>/<model>, in failover order
//! [models]
//! openai/gpt-5-mini
//! orcarouter/orcarouter/free
//! ```
//!
//! Models may contain slashes (`orcarouter/orcarouter/free`): only the first
//! segment names the provider, everything after it is the model sent upstream.

use std::fmt;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use thiserror::Error;

/// What a provider is useful for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Chat completions.
    Chat,
    /// `/v1/web_search`.
    Search,
    /// `/v1/web_fetch`.
    Fetch,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat",
            Self::Search => "search",
            Self::Fetch => "fetch",
        }
    }
}

/// API dialect a provider speaks. The chat dialects proxy chat requests; the
/// others back the router's own web endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApiKind {
    /// OpenAI-compatible `/v1/chat/completions` API (passed through).
    OpenAi,
    /// Anthropic `/v1/messages` API (translated to and from OpenAI's shape).
    Anthropic,
    /// DuckDuckGo's no-JS HTML results page (keyless).
    DuckDuckGo,
    /// MediaWiki search API, i.e. Wikipedia (keyless).
    Wikipedia,
    /// `r.jina.ai`: a URL as clean text (works without a key at 20 rpm).
    JinaReader,
    /// `s.jina.ai`: search results as clean text (needs a key).
    JinaSearch,
}

impl ApiKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OpenAi => "openai",
            Self::Anthropic => "anthropic",
            Self::DuckDuckGo => "duckduckgo",
            Self::Wikipedia => "wikipedia",
            Self::JinaReader => "jina",
            Self::JinaSearch => "jina-search",
        }
    }

    pub fn role(self) -> Role {
        match self {
            Self::OpenAi | Self::Anthropic => Role::Chat,
            Self::DuckDuckGo | Self::Wikipedia | Self::JinaSearch => Role::Search,
            Self::JinaReader => Role::Fetch,
        }
    }

    pub fn is_chat(self) -> bool {
        self.role() == Role::Chat
    }

    fn parse(raw: &str) -> Result<Self> {
        match raw.to_ascii_lowercase().as_str() {
            "openai" | "openai-compatible" | "oai" => Ok(Self::OpenAi),
            "anthropic" | "claude" => Ok(Self::Anthropic),
            "duckduckgo" | "ddg" | "ddg-html" => Ok(Self::DuckDuckGo),
            "wikipedia" | "wiki" | "mediawiki" => Ok(Self::Wikipedia),
            "jina" | "jina-reader" | "reader" => Ok(Self::JinaReader),
            "jina-search" | "jina_search" | "jina-search-api" => Ok(Self::JinaSearch),
            other => bail!(
                "unsupported api type `{other}` (supported: openai, anthropic, \
                 duckduckgo, wikipedia, jina, jina-search)"
            ),
        }
    }
}

impl fmt::Display for ApiKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One upstream API endpoint.
#[derive(Debug, Clone)]
pub struct Provider {
    pub api: ApiKind,
    pub base_url: String,
    pub shortname: String,
    /// Environment variable that holds the provider's API key.
    pub api_key_env: String,
    /// Key written directly in the config file (the 4th column), if any.
    pub api_key_inline: Option<String>,
}

impl Provider {
    /// The provider key: the inline value from the config when present,
    /// otherwise the environment variable. Providers without a key are still
    /// usable (e.g. a local Ollama server).
    pub fn api_key(&self) -> Option<String> {
        if let Some(key) = &self.api_key_inline {
            return Some(key.clone());
        }
        std::env::var(&self.api_key_env)
            .ok()
            .map(|key| key.trim().to_owned())
            .filter(|key| !key.is_empty())
    }

    /// Where the key came from, for logs (`config` or `$ENV_VAR`).
    pub fn key_source(&self) -> String {
        match self.api_key_inline {
            Some(_) => "config".to_owned(),
            None => format!("${}", self.api_key_env),
        }
    }
}

/// The environment variable used when the config line does not name one.
pub fn default_key_env(shortname: &str) -> String {
    let mut env = String::with_capacity(shortname.len() + 8);
    for ch in shortname.chars() {
        if ch.is_ascii_alphanumeric() {
            env.push(ch.to_ascii_uppercase());
        } else {
            env.push('_');
        }
    }
    env.push_str("_API_KEY");
    env
}

/// One entry from the `[models]` section.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub provider: String,
    /// Model name sent upstream, with the provider prefix removed.
    pub model: String,
    /// `provider/model`, as a client would name it.
    pub raw: String,
    line: usize,
}

/// A provider/model pair that a request may be sent to.
#[derive(Debug, Clone)]
pub struct Candidate<'a> {
    pub provider: &'a Provider,
    /// Index of `provider` in [`Config::providers`], for per-provider state.
    pub index: usize,
    pub model: String,
}

#[derive(Debug, Error)]
pub enum PlanError {
    #[error("no models configured in the [models] section")]
    NoModels,
    #[error("model `{requested}` is not configured (configured: {available})")]
    UnknownModel {
        requested: String,
        available: String,
    },
}

#[derive(Debug, Clone)]
pub struct Config {
    pub providers: Vec<Provider>,
    pub models: Vec<ModelEntry>,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading config file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing config file {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut providers = Vec::new();
        let mut models: Vec<ModelEntry> = Vec::new();
        let mut section: Option<Section> = None;

        for (idx, raw) in text.lines().enumerate() {
            let line_no = idx + 1;
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            // `${VAR}` and `${VAR:-default}` let one config file serve many
            // machines without ever holding a secret.
            let expanded = expand_env(line).with_context(|| format!("line {line_no}"))?;
            let line = expanded.as_str();

            if let Some(name) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                section = Some(match name.trim().to_ascii_lowercase().as_str() {
                    "providers" => Section::Providers,
                    "models" => Section::Models,
                    other => bail!(
                        "line {line_no}: unknown section `[{other}]` \
                         (expected [providers] or [models])"
                    ),
                });
                continue;
            }

            match section {
                Some(Section::Providers) => {
                    let provider = parse_provider(line)
                        .with_context(|| format!("line {line_no}: invalid provider entry"))?;
                    if let Some(dup) = providers
                        .iter()
                        .find(|existing: &&Provider| existing.shortname == provider.shortname)
                    {
                        bail!(
                            "line {line_no}: duplicate provider shortname `{}` \
                             (already defined as {})",
                            provider.shortname,
                            dup.base_url
                        );
                    }
                    providers.push(provider);
                }
                Some(Section::Models) => {
                    let entry = parse_model(line, line_no)?;
                    // Duplicates are harmless, keep only the first occurrence.
                    if !models.iter().any(|existing| existing.raw == entry.raw) {
                        models.push(entry);
                    }
                }
                None => bail!(
                    "line {line_no}: entries must follow a `[providers]` or `[models]` header"
                ),
            }
        }

        if providers.is_empty() {
            bail!("no providers configured in the [providers] section");
        }
        for model in &models {
            match providers.iter().find(|p| p.shortname == model.provider) {
                None => bail!(
                    "line {}: model `{}` refers to unknown provider `{}`",
                    model.line,
                    model.raw,
                    model.provider
                ),
                Some(provider) if !provider.api.is_chat() => bail!(
                    "line {}: model `{}` refers to `{}`, which is a {} provider \
                     (only openai/anthropic providers serve models)",
                    model.line,
                    model.raw,
                    model.provider,
                    provider.api.role().as_str()
                ),
                Some(_) => {}
            }
        }

        Ok(Self { providers, models })
    }

    /// Chat providers with their index, in file order.
    pub fn chat_providers(&self) -> impl Iterator<Item = (usize, &Provider)> {
        self.providers
            .iter()
            .enumerate()
            .filter(|(_, provider)| provider.api.is_chat())
    }

    /// Web search providers with their index, in failover order.
    pub fn search_providers(&self) -> impl Iterator<Item = (usize, &Provider)> {
        self.providers
            .iter()
            .enumerate()
            .filter(|(_, provider)| provider.api.role() == Role::Search)
    }

    /// Page-fetch providers with their index, in failover order.
    pub fn fetch_providers(&self) -> impl Iterator<Item = (usize, &Provider)> {
        self.providers
            .iter()
            .enumerate()
            .filter(|(_, provider)| provider.api.role() == Role::Fetch)
    }

    pub fn has_search(&self) -> bool {
        self.search_providers().next().is_some()
    }

    pub fn has_fetch(&self) -> bool {
        self.fetch_providers().next().is_some()
    }

    pub fn provider(&self, shortname: &str) -> Option<&Provider> {
        self.providers.iter().find(|p| p.shortname == shortname)
    }

    /// Providers to try for a request, in order.
    ///
    /// * a model configured in `[models]` starts the list at that entry, then
    ///   keeps falling through the rest of the file order;
    /// * a nameless/`auto` model just uses the file order;
    /// * an unconfigured `provider/model` is sent to that provider first, then
    ///   falls through the configured models.
    pub fn candidates(&self, requested: Option<&str>) -> Result<Vec<Candidate<'_>>, PlanError> {
        if self.models.is_empty() {
            return Err(PlanError::NoModels);
        }

        let requested = requested
            .map(str::trim)
            .filter(|m| !m.is_empty() && !matches!(*m, "auto" | "default" | "router"));

        let mut candidates = Vec::new();
        match requested {
            None => {
                for entry in &self.models {
                    self.push_candidate(&mut candidates, &entry.provider, entry.model.clone());
                }
            }
            Some(name) => {
                if let Some(start) = self.models.iter().position(|m| m.raw == name) {
                    for entry in self.models[start..].iter().chain(&self.models[..start]) {
                        self.push_candidate(&mut candidates, &entry.provider, entry.model.clone());
                    }
                } else if let Some((shortname, model)) = name.split_once('/') {
                    if self.provider(shortname).is_none() {
                        return Err(self.unknown_model(name));
                    }
                    self.push_candidate(&mut candidates, shortname, model.to_owned());
                    for entry in &self.models {
                        self.push_candidate(&mut candidates, &entry.provider, entry.model.clone());
                    }
                } else {
                    return Err(self.unknown_model(name));
                }
            }
        }

        if candidates.is_empty() {
            return Err(PlanError::NoModels);
        }
        Ok(candidates)
    }

    fn push_candidate<'a>(
        &'a self,
        candidates: &mut Vec<Candidate<'a>>,
        shortname: &str,
        model: String,
    ) {
        let Some(index) = self.providers.iter().position(|p| p.shortname == shortname) else {
            return;
        };
        // Web providers have no models; never route chat to them.
        if !self.providers[index].api.is_chat() {
            return;
        }
        if candidates.iter().any(|c| c.index == index && c.model == model) {
            return;
        }
        candidates.push(Candidate {
            provider: &self.providers[index],
            index,
            model,
        });
    }

    fn unknown_model(&self, requested: &str) -> PlanError {
        let available = self
            .models
            .iter()
            .map(|m| m.raw.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        PlanError::UnknownModel {
            requested: requested.to_owned(),
            available,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum Section {
    Providers,
    Models,
}

/// Substitute `$VAR`, `${VAR}` and `${VAR:-default}`. `$$` yields a literal `$`.
fn expand_env(line: &str) -> Result<String> {
    let mut out = String::with_capacity(line.len());
    let mut rest = line;

    while let Some(pos) = rest.find('$') {
        out.push_str(&rest[..pos]);
        let after = &rest[pos + 1..];

        if let Some(escaped) = after.strip_prefix('$') {
            out.push('$');
            rest = escaped;
            continue;
        }

        if let Some(braced) = after.strip_prefix('{') {
            let end = braced
                .find('}')
                .context("unterminated `${` (missing `}`)")?;
            let expr = &braced[..end];
            let (name, default) = match expr.split_once(":-") {
                Some((name, default)) => (name, Some(default)),
                None => (expr, None),
            };
            if name.is_empty() {
                bail!("empty variable name in `${{{expr}}}`");
            }
            let value = lookup_env(name, default)?;
            out.push_str(&value);
            rest = &braced[end + 1..];
            continue;
        }

        let end = after
            .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
            .unwrap_or(after.len());
        if end == 0 {
            out.push('$');
            rest = after;
            continue;
        }
        out.push_str(&lookup_env(&after[..end], None)?);
        rest = &after[end..];
    }

    out.push_str(rest);
    Ok(out)
}

fn lookup_env(name: &str, default: Option<&str>) -> Result<String> {
    match std::env::var(name) {
        Ok(value) if !value.is_empty() => Ok(value),
        _ => match default {
            Some(default) => Ok(default.to_owned()),
            None => bail!("environment variable `{name}` is not set"),
        },
    }
}

fn parse_provider(line: &str) -> Result<Provider> {
    let fields: Vec<&str> = line.split(',').map(str::trim).collect();
    let (api, base_url, shortname, key_field) = match fields.as_slice() {
        [api, base_url, shortname] => (*api, *base_url, *shortname, None),
        [api, base_url, shortname, key] => (*api, *base_url, *shortname, Some(*key)),
        _ => {
            bail!(
                "expected `<api type>,<base url>,<shortname>[,<api key | env var name>]`, \
                 got `{line}`"
            )
        }
    };

    if base_url.is_empty() || shortname.is_empty() {
        bail!("base url and shortname must not be empty in `{line}`");
    }
    if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
        bail!("base url `{base_url}` must start with http:// or https://");
    }

    let (api_key_inline, api_key_env) = match key_field {
        None | Some("") => (None, default_key_env(shortname)),
        // Explicit prefixes win over the guess below.
        Some(raw) if raw.starts_with("env:") => {
            let name = raw.trim_start_matches("env:").trim();
            if name.is_empty() {
                bail!("`env:` needs a variable name in `{line}`");
            }
            (None, name.to_owned())
        }
        Some(raw) if raw.starts_with("key:") => {
            let key = raw.trim_start_matches("key:").trim();
            if key.is_empty() {
                bail!("`key:` needs a value in `{line}`");
            }
            (Some(key.to_owned()), default_key_env(shortname))
        }
        // A bare `UPPER_SNAKE` token is an environment variable name; anything
        // else (api keys have lowercase, dashes, dots or spaces) is the key.
        Some(raw) if looks_like_env_name(raw) => (None, raw.to_owned()),
        Some(raw) => (Some(raw.to_owned()), default_key_env(shortname)),
    };

    Ok(Provider {
        api: ApiKind::parse(api)?,
        base_url: base_url.trim_end_matches('/').to_owned(),
        shortname: shortname.to_owned(),
        api_key_env,
        api_key_inline,
    })
}

/// `OPENAI_API_KEY`, `OPENROUTER_KEY`, ... — never a real key's shape.
fn looks_like_env_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .chars()
            .all(|ch| ch.is_ascii_uppercase() || ch.is_ascii_digit() || ch == '_')
        && value.chars().any(|ch| ch.is_ascii_uppercase())
}

fn parse_model(line: &str, line_no: usize) -> Result<ModelEntry> {
    let Some((provider, model)) = line.split_once('/') else {
        bail!("line {line_no}: model `{line}` must be written as <provider>/<model>");
    };
    let (provider, model) = (provider.trim(), model.trim());
    if provider.is_empty() || model.is_empty() {
        bail!("line {line_no}: model `{line}` must be written as <provider>/<model>");
    }
    Ok(ModelEntry {
        provider: provider.to_owned(),
        model: model.to_owned(),
        raw: format!("{provider}/{model}"),
        line: line_no,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
# providers
[providers]
openai,https://api.openai.com/v1,openai
openai,https://openrouter.ai/api/v1,orcarouter,OPENROUTER_KEY
openai,https://api.groq.com/openai/v1,groq,gsk-abc123
openai,https://api.x.ai/v1,xai,env:XAI_KEY
openai,https://api.example.com/v1,example,key:UPPERCASEKEY

[models]
openai/gpt-5-mini
orcarouter/orcarouter/free
"#;

    #[test]
    fn parses_providers_and_models() {
        let config = Config::parse(SAMPLE).unwrap();
        assert_eq!(config.providers.len(), 5);
        assert_eq!(config.providers[1].api_key_env, "OPENROUTER_KEY");
        assert_eq!(config.providers[0].api_key_env, "OPENAI_API_KEY");
        // Bare non-env values are literal keys, `key:`/`env:` are explicit.
        assert_eq!(config.providers[2].api_key().as_deref(), Some("gsk-abc123"));
        assert_eq!(config.providers[3].api_key_env, "XAI_KEY");
        assert_eq!(
            config.providers[4].api_key_inline.as_deref(),
            Some("UPPERCASEKEY")
        );
        assert_eq!(config.providers[0].key_source(), "$OPENAI_API_KEY");
        assert_eq!(config.providers[2].key_source(), "config");
        assert_eq!(config.models.len(), 2);
        // only the first segment names the provider
        assert_eq!(config.models[1].provider, "orcarouter");
        assert_eq!(config.models[1].model, "orcarouter/free");
        assert_eq!(config.models[1].raw, "orcarouter/orcarouter/free");
    }

    #[test]
    fn model_failover_starts_at_the_requested_entry() {
        let config = Config::parse(SAMPLE).unwrap();
        let candidates = config
            .candidates(Some("orcarouter/orcarouter/free"))
            .unwrap();
        let plan: Vec<_> = candidates
            .iter()
            .map(|c| format!("{}/{}", c.provider.shortname, c.model))
            .collect();
        assert_eq!(
            plan,
            vec!["orcarouter/orcarouter/free", "openai/gpt-5-mini",]
        );
    }

    #[test]
    fn auto_uses_file_order() {
        let config = Config::parse(SAMPLE).unwrap();
        let plan: Vec<_> = config
            .candidates(Some("auto"))
            .unwrap()
            .iter()
            .map(|c| c.provider.shortname.clone())
            .collect();
        assert_eq!(plan, vec!["openai", "orcarouter"]);
    }

    #[test]
    fn unconfigured_provider_prefixed_model_is_still_routable() {
        let config = Config::parse(SAMPLE).unwrap();
        let plan: Vec<_> = config
            .candidates(Some("openai/gpt-4o"))
            .unwrap()
            .iter()
            .map(|c| format!("{}/{}", c.provider.shortname, c.model))
            .collect();
        assert_eq!(
            plan,
            vec![
                "openai/gpt-4o",
                "openai/gpt-5-mini",
                "orcarouter/orcarouter/free",
            ]
        );
    }

    #[test]
    fn unknown_model_is_rejected() {
        let config = Config::parse(SAMPLE).unwrap();
        let err = config.candidates(Some("nope/gpt")).unwrap_err();
        assert!(err.to_string().contains("is not configured"), "{err}");
        assert!(err.to_string().contains("openai/gpt-5-mini"), "{err}");
    }

    #[test]
    fn unknown_provider_in_models_is_rejected() {
        let err = Config::parse("[providers]\nopenai,https://x.dev/v1,openai\n[models]\nnope/x\n")
            .unwrap_err();
        assert!(err.to_string().contains("unknown provider `nope`"));
    }

    #[test]
    fn duplicate_shortnames_are_rejected() {
        let err = Config::parse(
            "[providers]\nopenai,https://a.dev/v1,dup\nopenai,https://b.dev/v1,dup\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("duplicate provider shortname"));
    }

    #[test]
    fn entries_before_a_section_are_rejected() {
        let err = Config::parse("openai,https://a.dev/v1,openai\n").unwrap_err();
        assert!(err.to_string().contains("[providers]"));
    }

    #[test]
    fn expands_env_vars_with_defaults() {
        let config = Config::parse(
            "[providers]\nopenai,${ROUTER_TEST_UNSET_HOST:-https://api.openai.com/v1},openai\n\
             [models]\nopenai/gpt-test\n",
        )
        .unwrap();
        assert_eq!(config.providers[0].base_url, "https://api.openai.com/v1");
    }

    #[test]
    fn rejects_unset_env_vars() {
        let err = Config::parse(
            "[providers]\nopenai,${ROUTER_TEST_DEFINITELY_UNSET}/v1,openai\n",
        )
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("ROUTER_TEST_DEFINITELY_UNSET"),
            "{err:#}"
        );
    }

    #[test]
    fn dollar_escape_stays_literal() {
        let config = Config::parse(
            "[providers]\nopenai,https://x.dev/v1,openai\n[models]\nopenai/model$$x\n",
        )
        .unwrap();
        assert_eq!(config.models[0].model, "model$x");
    }

    #[test]
    fn parses_anthropic_providers() {
        let config = Config::parse(
            "[providers]\nanthropic,https://api.anthropic.com,claude\n\
             [models]\nclaude/claude-sonnet-4-5\n",
        )
        .unwrap();
        assert_eq!(config.providers[0].api, ApiKind::Anthropic);
        assert_eq!(config.providers[0].api_key_env, "CLAUDE_API_KEY");
    }

    #[test]
    fn web_providers_are_separate_from_chat_providers() {
        let config = Config::parse(
            "[providers]\n\
             openai,https://api.openai.com/v1,openai\n\
             duckduckgo,https://html.duckduckgo.com,ddg\n\
             wikipedia,https://en.wikipedia.org,wiki\n\
             jina,https://r.jina.ai,jina\n\
             jina-search,https://s.jina.ai,jina-search,env:JINA_API_KEY\n\
             [models]\nopenai/gpt-5-mini\n",
        )
        .unwrap();
        let chat: Vec<_> = config.chat_providers().map(|(_, p)| p.shortname.as_str()).collect();
        assert_eq!(chat, vec!["openai"]);
        let search: Vec<_> = config
            .search_providers()
            .map(|(_, p)| p.shortname.as_str())
            .collect();
        assert_eq!(search, vec!["ddg", "wiki", "jina-search"]);
        let fetch: Vec<_> = config
            .fetch_providers()
            .map(|(_, p)| p.shortname.as_str())
            .collect();
        assert_eq!(fetch, vec!["jina"]);
        assert!(config.has_search() && config.has_fetch());
    }

    #[test]
    fn models_cannot_point_at_a_web_provider() {
        let err = Config::parse(
            "[providers]\nduckduckgo,https://html.duckduckgo.com,ddg\n[models]\nddg/x\n",
        )
        .unwrap_err();
        assert!(err.to_string().contains("which is a search provider"), "{err}");
    }

    #[test]
    fn key_env_names_are_sanitized() {
        assert_eq!(default_key_env("orcarouter"), "ORCAROUTER_API_KEY");
        assert_eq!(default_key_env("my-proxy.2"), "MY_PROXY_2_API_KEY");
    }
}

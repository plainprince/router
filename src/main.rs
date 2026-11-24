
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use router::config::Config;
use router::router::{Options, Router};
use router::server::{self, AppState};
use tracing_subscriber::EnvFilter;

const DEFAULT_CONFIG: &str = "router.conf";
const DEFAULT_ADDR: &str = "127.0.0.1:8354";
const DEFAULT_ATTEMPT_TIMEOUT_SECS: u64 = 45;
/// How long a failed model is skipped before the list reaches it again.
const DEFAULT_MODEL_TIMEOUT_SECS: u64 = 20;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("router=info,info")),
        )
        .with_target(false)
        .init();

    let mut args = std::env::args().skip(1);
    let config_path = match args.next().as_deref() {
        Some("-h" | "--help") => {
            print_help();
            return Ok(());
        }
        Some(flag) if flag.starts_with('-') => bail!("unknown option `{flag}` (try --help)"),
        Some(path) => PathBuf::from(path),
        None => std::env::var_os("ROUTER_CONFIG")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG)),
    };

    let config = Config::load(&config_path)?;
    let options = Options {
        attempt_timeout: env_secs("ROUTER_ATTEMPT_TIMEOUT_SECS", DEFAULT_ATTEMPT_TIMEOUT_SECS),
        model_timeout: env_secs("ROUTER_MODEL_TIMEOUT_SECS", DEFAULT_MODEL_TIMEOUT_SECS),
    };
    let addr = std::env::var("ROUTER_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_owned());
    let token = std::env::var("ROUTER_TOKEN")
        .ok()
        .filter(|token| !token.is_empty())
        .map(Arc::from);

    log_config(&config, options);

    let router = Router::new(config, options)?;
    let state = AppState {
        router: Arc::new(router),
        token,
    };

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .with_context(|| format!("binding {addr}"))?;
    tracing::info!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, server::app(state))
        .with_graceful_shutdown(shutdown_signal())
        .await
        .context("serving requests")?;
    Ok(())
}

/// Failover order, and which providers actually have a key configured.
fn log_config(config: &Config, options: Options) {
    for provider in &config.providers {
        match provider.api_key() {
            Some(_) => tracing::info!(
                provider = %provider.shortname,
                api = %provider.api,
                base_url = %provider.base_url,
                "provider ready (key from {})",
                provider.key_source()
            ),
            None => tracing::warn!(
                provider = %provider.shortname,
                api = %provider.api,
                base_url = %provider.base_url,
                "provider has no key ({} is unset), sending requests unauthenticated",
                provider.api_key_env
            ),
        }
    }
    let order: Vec<&str> = config.models.iter().map(|m| m.raw.as_str()).collect();
    tracing::info!(
        failover_order = %order.join(" -> "),
        attempt_timeout_secs = options.attempt_timeout.as_secs(),
        model_timeout_secs = options.model_timeout.as_secs(),
        "models loaded"
    );
}

fn env_secs(name: &str, default: u64) -> Duration {
    match std::env::var(name) {
        Ok(value) => match value.trim().parse::<u64>() {
            Ok(secs) if secs > 0 => Duration::from_secs(secs),
            _ => {
                tracing::warn!("ignoring invalid {name}=`{value}`, using {default}s");
                Duration::from_secs(default)
            }
        },
        Err(_) => Duration::from_secs(default),
    }
}

async fn shutdown_signal() {
    if let Err(err) = tokio::signal::ctrl_c().await {
        tracing::error!("could not listen for shutdown signal: {err}");
    }
    tracing::info!("shutting down");
}

fn print_help() {
    println!(
        "\
router - OpenAI-compatible failover proxy

USAGE:
    router [CONFIG]

ARGS:
    CONFIG    Path to the config file (default: $ROUTER_CONFIG or ./{DEFAULT_CONFIG})

ENVIRONMENT:
    ROUTER_ADDR                    Listen address (default: {DEFAULT_ADDR})
    ROUTER_TOKEN                   Require this bearer token from clients
    ROUTER_ATTEMPT_TIMEOUT_SECS    Time to wait for a provider's answer (default: {DEFAULT_ATTEMPT_TIMEOUT_SECS})
    ROUTER_MODEL_TIMEOUT_SECS      How long a failed model is skipped (default: {DEFAULT_MODEL_TIMEOUT_SECS})
    RUST_LOG                       Log filter (default: router=info,info)

Provider keys are read from `<SHORTNAME>_API_KEY`, or the env var named in the
fourth column of the provider line. They never belong in the config file; any
`$VAR` or `${{VAR:-default}}` in the file is expanded from the environment."
    );
}

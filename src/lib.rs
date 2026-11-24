//! Provider failover in front of OpenAI- and Anthropic-compatible APIs, plus the
//! web tools (`web_search`, `web_fetch`) an agent can call.
//!
//! The binary is a thin wrapper; embedding applications build a [`config::Config`],
//! a [`router::Router`], and serve [`server::app`] themselves.

pub mod agent;
pub mod anthropic;
pub mod config;
pub mod context;
pub mod router;
pub mod runtime;
pub mod server;
pub mod web;

# router

OpenAI-compatible failover proxy. Clients point their base URL at this service;
it picks the first configured provider that answers, across OpenAI-compatible
and Anthropic providers.

The optional `../improving` service layers self-improvement passes on top of
the router (router on `8354`, improver on `8355`).

## Run

```bash
cp router.conf.example router.conf
OPENAI_API_KEY=sk-... ANTHROPIC_API_KEY=sk-ant-... RUST_LOG=router=info cargo run
```

```
GET  /health           providers, key presence, failover order, web tools
GET  /v1/health        the same, for clients that probe `/v1/health` (never proxied)
GET  /v1/models        configured `provider/model` ids
POST /v1/web_search    {"query": "…", "limit": 6} → results + text
POST /v1/web_fetch     {"url": "https://…"}        → title + text
POST /v1/chat/completions  proxied with failover; `"web": true` turns it into
                           an agent run that may call web_search/web_fetch
*    /v1/*            proxied, with failover
```

## Web tools

Providers whose api type is a web dialect back the two endpoints above instead
of chat. Search fails over in file order; a provider that is rate limited,
errors, or returns a page with no parseable results is skipped immediately.

| api type      | Role   | Key | Notes |
| ------------- | ------ | --- | ----- |
| `duckduckgo`  | search | none | No-JS HTML results; no quota, occasional bot check |
| `wikipedia`   | search | none | MediaWiki search API; stable, factual |
| `jina-search` | search | free | `s.jina.ai`, text results, ~1000 free searches |
| `jina`        | fetch  | none | `r.jina.ai`, page → clean text, 20 rpm keyless |

With any of them configured, a chat request carrying `"web": true` runs a
small agent loop: the router advertises `web_search`/`web_fetch`, executes the
calls, feeds the results back as `tool` messages, and returns the final answer
(up to two tool rounds). Tool calls the router does not own are handed back to
the client untouched. The web sources used are returned as `"web"` in the JSON
response and as a choice-less SSE chunk when streaming, so a UI can cite them.

```bash
curl localhost:8354/v1/web_search -H 'content-type: application/json' \
  -d '{"query":"rust axum routing","limit":3}'
```

| Environment                          | Default          | Meaning                                              |
| ------------------------------------ | ---------------- | ---------------------------------------------------- |
| `ROUTER_ADDR`                        | `127.0.0.1:8354` | Listen address                                        |
| `ROUTER_TOKEN`                       | unset            | Require this bearer token from clients                |
| `ROUTER_ATTEMPT_TIMEOUT_SECS`        | `45`             | Time to wait for a provider's answer                  |
| `ROUTER_MODEL_TIMEOUT_SECS`          | `20`             | How long a failed model is skipped                    |
| `ROUTER_CONFIG`                      | `router.conf`    | Config path (or pass it as the first argument)        |
| `RUST_LOG`                           | `router=info`    | Log filter                                            |

```bash
curl localhost:8354/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hi"}],"stream":true}' \
  -D -   # x-router-provider / x-router-model say who served it
```

### Runtime context

Models have no clock, so a prompt without a date makes them guess the year
from their training data. Before every chat completion the router merges a short
block into the leading system message (or adds one) naming the current UTC date
and time with the weekday, and the `provider/model` that is serving this attempt:

```
[runtime] Environment facts for this conversation, which you must treat as current and authoritative:
- Current date and time: 2026-11-23 15:53:27 Monday UTC
- Model answering this request: orcarouter/z-ai/glm-5.3-flash-free
- Requests are routed through a local model router with automatic failover; …
```

It is added per attempt because the model changes on failover, only on
`/v1/chat/completions`, and never twice. It is also added to `"web": true` agent
calls, so every mode (direct proxy, agent, and the improvement layers) sees it.

### Context limit

Send `"max_context": <tokens>` in a chat request to cap the whole prompt. Before
any provider is tried the router removes the field, drops the oldest non-system
turns until the messages and `tools` fit, trims a single oversized message
(keeping the end of ordinary turns), and clamps `max_tokens` to the room that is
left. Tokens are estimated at four characters per token, which is conservative.

The limit also applies to `"web": true` agent runs (each internal call is
trimmed again as tool results accumulate). When something was dropped the
response carries `x-router-context-dropped` (estimated tokens removed) and
`x-router-context-messages` (turns dropped).

## Config

`router.conf` is gitignored; start from [`router.conf.example`](router.conf.example).

- `[providers]`: `<api type>,<base url>,<shortname>[,<api key | env var name>]`
- `[models]`: `<provider>/<model>`, in failover order
- `#` starts a comment; `$VAR` and `${VAR:-default}` expand from the environment
- The optional 4th column carries the key itself, so keys can live next to
  their provider:
  ```ini
  openai,https://api.orcarouter.ai/v1,orcarouter,sk-orca-...
  ```
  A bare `UPPER_SNAKE` token in that column is treated as an **environment
  variable name** (`OPENROUTER_KEY`), which keeps keys out of the file. The
  explicit forms are always unambiguous: `key:<literal>` and `env:<VAR>`.
  Without a 4th column the variable defaults to `<SHORTNAME>_API_KEY`.
  A provider with no key at all is sent unauthenticated (handy for a local
  server).
- `router.conf` is gitignored; keep it that way when it contains keys, and it
  must not be world-readable (`chmod 600 router.conf`).

`api type` is `openai` or `anthropic`. Only the first `/` in a model names the
provider, so gateway routes keep their own slashes (`orcarouter/orcarouter/auto`
is sent to that provider as `orcarouter/auto`).

## Routing

A request's `model` selects where the attempt list starts, then it falls through
the rest of the config order:

| requested `model`            | attempts                                  |
| ---------------------------- | ----------------------------------------- |
| `auto` / `default` / absent  | config order                              |
| a `[models]` entry           | that entry, then the rest of config order |
| `knownprovider/some-model`   | that pair, then all `[models]` entries    |
| anything else                | `400` listing the configured models       |

The `[models]` list is walked **in exactly its written order, one model after
the other**. Only `200 OK` and `304 Not Modified` count as an answer: **every
other outcome is a failed attempt** — including `400` (a provider that rejects a
prompt, tool schema or model name says nothing about the next one), connection
errors, a provider that sends no headers within `ROUTER_ATTEMPT_TIMEOUT_SECS`,
and a body that arrives too late or cannot be read.

A failed model is skipped for `ROUTER_MODEL_TIMEOUT_SECS` (20s) and the walk
starts again from the **top of the list**, skipping every model still on
timeout. When the end of the list is reached (every model tried or skipped), all
timeouts are cleared and the list is walked again; after two repeats (three
walks in total) the request fails. The response is `502` with an `attempts`
array naming each provider, its model and its final error (one entry per model).

There is no overall deadline on a request: the walk always runs its course.
`ROUTER_ATTEMPT_TIMEOUT_SECS` bounds each individual attempt (that bound is what
makes a provider that sends no headers count as failed), not the failover as a
whole.

### Concurrency

Requests are independent; there is no queue and no shared lock. Failure state
is kept per `provider/model`, so while one request restarts its walk, concurrent
requests skip that model immediately and keep using its neighbours on the same
provider. Free tiers limit individual models, which is why the timeout is not
tied to the provider.

### Streaming

Responses stream through as they arrive. For `anthropic` providers the
`/v1/messages` SSE events are translated into OpenAI `chat.completion.chunk`
events on the fly, so tokens are still delivered incrementally.

## Anthropic translation

Clients speak OpenAI. For `anthropic` providers the router maps:

- `system`/`developer` messages → top-level `system`
- text, `image_url` (data URLs and remote URLs), `tool_calls` → `tool_use`
- `role: "tool"` messages → `tool_result` blocks, adjacent turns merged
- `max_tokens` (default 8192), `temperature`, `top_p`, `stop`, `stream`,
  `tools`/`tool_choice`
- responses back to `chat.completion` with `tool_calls`, `finish_reason` and
  `usage`, and stream errors as an OpenAI-style `error` chunk

Only `/v1/chat/completions` is translated; other paths fail over past anthropic
providers.

## Embedding

The crate is also a library: build a `config::Config`, a `router::Router`, and
serve `server::app(AppState { router, token })` from your own process. The
`frontend` crate does exactly that, on an ephemeral loopback port.

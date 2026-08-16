# Running agentic-api without a cluster

Session compaction on a laptop. Two containers, any OpenAI-compatible backend,
no Kubernetes and no GPU.

agentic-api remembers the history it has already seen for a session, stores a
compacted stand-in, and substitutes it on later turns — so the prompt stops
growing even though the client keeps resending the whole conversation.

## 1. Compactor config

context-guru ships **no** config inside the image — you must supply one. Start
with the deterministic mode, which needs no LLM of its own:

```bash
mkdir -p configs && cat > configs/toon.yaml <<'YAML'
# Deterministic reformatter: re-encodes uniform JSON arrays in tool messages as
# TOON — field names once, then one row per element. No LLM, nothing stored.
pipeline: [format, toon]
components:
  format:
    min_tokens: 50
  toon:
    min_tokens: 50
store:
  enabled: false
YAML
```

> **The config path is fixed by the image's own command** (`--config
> /app/configs/toon.yaml`). Setting a `CONFIG` environment variable does
> **not** work — the flag wins and the variable is ignored. Either keep your
> chosen config at that exact path, or override the command as shown in §5.

## 2. Run both containers

```bash
cat > compose.yaml <<'YAML'
services:
  context-guru:
    image: ghcr.io/ronenkat/context-guru-proxy
    volumes: [./configs:/app/configs:ro]
    # The mode is chosen by this flag, not by an env var.
    command: ["--config", "/app/configs/toon.yaml"]

  agentic-api:
    image: ghcr.io/mohammad-nassar10/agentic-api:thresholds
    ports: ["9000:9000"]
    depends_on: [context-guru]
    environment:
      LLM_API_BASE: https://api.openai.com     # any OpenAI-compatible endpoint
      OPENAI_API_KEY: ${OPENAI_API_KEY}
      COMPACTION_ADDRESS: http://context-guru:4000/compact
      SKIP_LLM_READY_CHECK: "true"
      DATABASE_URL: sqlite://./agentic_api.db
      RUST_LOG: agentic_server=debug
YAML

OPENAI_API_KEY=sk-... docker compose up
```

Point `LLM_API_BASE` at whatever you have — OpenAI, a local vLLM
(`http://host.docker.internal:8000`), Ollama, anything speaking
`/v1/chat/completions`.

Check it:

```bash
curl -s localhost:9000/health          # 200
```

## 3. Send two turns

Compaction pays off on the **second** turn — the first one is what gets stored.

```bash
SID=demo-1
curl -s localhost:9000/v1/chat/completions -H 'content-type: application/json' \
  -H "x-session-id: $SID" \
  -d '{"model":"gpt-4o-mini","messages":[{"role":"user","content":"hello"}]}' \
  | grep -o '"prompt_tokens":[0-9]*'

sleep 3   # the fold runs in the background, after the reply

curl -s localhost:9000/v1/chat/completions -H 'content-type: application/json' \
  -H "x-session-id: $SID" \
  -d '{"model":"gpt-4o-mini","messages":[
        {"role":"user","content":"hello"},
        {"role":"assistant","content":"Hi!"},
        {"role":"user","content":"and again"}]}' \
  | grep -o '"prompt_tokens":[0-9]*'
```

`docker compose logs agentic-api` shows what happened:

```
folded session prefix      session=demo-1 replaced=1 stored=1
substituted stored prefix  session=demo-1 replaced=1 upstream_messages=3 client_messages=3
```

Without `x-session-id` nothing is stored and nothing is substituted — that's your
control case.

> The compactor only rewrites JSON arrays inside `tool` messages, and TOON's
> header costs more than it saves below ~10 records. For a visible saving, send a
> `tool` message containing a 20+ element array.

## 4. Configuration

| variable | default | purpose |
|---|---|---|
| `LLM_API_BASE` | — | upstream OpenAI-compatible endpoint (**required**) |
| `OPENAI_API_KEY` | — | forwarded upstream when the request has no `Authorization` |
| `COMPACTION_ADDRESS` | unset | compactor URL; unset disables compaction entirely |
| `SKIP_LLM_READY_CHECK` | `false` | set `true` if the backend has no `/v1/models` |
| `DATABASE_URL` | `sqlite://./agentic_api.db` | in-container by default — restarts lose all sessions |
| `GATEWAY_PORT` | `9000` | listen port |
| `RUST_LOG` | — | `agentic_server=debug` to see folds and substitutions |

### Compaction thresholds (optional)

By default **every turn is compacted**. When the backend is llm-d with the
`metrics-to-headers` plugin, it reports its own load on each response and these
limit compaction to when it is actually busy:

| variable | unit |
|---|---|
| `COMPACTION_KV_CACHE_THRESHOLD` | fraction `0`–`1` |
| `COMPACTION_WAITING_QUEUE_THRESHOLD` | requests |
| `COMPACTION_RUNNING_REQUESTS_THRESHOLD` | requests |
| `COMPACTION_MAX_METRICS_AGE_MS` | milliseconds (staleness guard) |

Any one being met triggers compaction — a full cache and a long queue each
justify it alone. Anything uncertain compacts: no thresholds set, no metrics
headers (any non-llm-d backend), or readings too old. A malformed value stops the
server at startup rather than being silently ignored.

These do nothing against OpenAI or a plain vLLM — those backends send no metrics
headers, so every turn compacts as usual.

## 5. Configuring the compactor

Three modes. Only the first needs no model of its own.

| mode | what it does | needs an LLM | changes message count |
|---|---|---|---|
| `toon` | re-encodes JSON arrays in `tool` messages as TOON | no | no |
| `extract` | an LLM writes a filter that deletes irrelevant lines | yes | no |
| `summarize` | an LLM compresses the middle of the transcript | yes | **yes** |

Switch by pointing the command at a different file:

```yaml
    command: ["--config", "/app/configs/summarize.yaml"]
```

### toon — deterministic

```yaml
pipeline: [format, toon]
components:
  format: {min_tokens: 50}
  toon:   {min_tokens: 50}
store:
  enabled: false
```

`min_tokens` is the smallest payload worth touching. Lower it to ~10 if your test
arrays are small; below ~10 records TOON's header costs more than it saves, so
the result can be *larger* than the input.

### summarize — LLM, collapses many messages into one

```yaml
pipeline: [summarize]
components:
  summarize:
    summary_level: regular     # concise | regular | highly_detailed
    keep_last: 3               # turns kept verbatim at the tail
    min_tokens: 500            # smallest span worth summarizing
    resummarize_tokens: 6000   # roll the summary forward past this much new tail
    marker_mode: "off"         # irreversible: no markers, nothing stashed
    trigger:
      min_messages: 12         # ← lower these two for a small demo
      min_request_tokens: 40000
    model:
      source: config
      provider: openai
      base_url: https://api.openai.com    # NO trailing /v1 — see below
      model: gpt-4o-mini
      api_key: ""              # empty falls back to OPENAI_API_KEY
store:
  enabled: false
```

### extract — LLM, deletes irrelevant lines from large tool output

Same `model:` block; the triggers are `min_request_tokens: 20000` and
`min_output_tokens: 2000`.

### Three ways an LLM mode silently does nothing

**A trailing `/v1` on `base_url`.** context-guru appends the full
`/v1/chat/completions` itself, so `.../v1` becomes `/v1/v1/chat/completions`,
matches nothing, and the input comes back unchanged in milliseconds with no
error. A working call takes seconds.

**Triggers left at their defaults.** They assume 20k–40k token payloads. A small
test conversation is skipped and returned unchanged — identical in appearance to
a broken deployment. Drop them to a few hundred while experimenting.

**A reasoning model with thinking enabled.** Qwen3 and similar emit a `<think>`
block before the answer; asked to summarize a short passage one produced 400
completion tokens of reasoning against a 52-token input. The result is longer
than the text it would replace, so context-guru discards it. On vLLM, disable it
server-side with
`--default-chat-template-kwargs={"enable_thinking": false}`. Against OpenAI's
non-reasoning models this does not arise.

### Reading the log

```
component=summarize tokens.before=407 tokens.after=161 tokens.saved=246 reverted=false   ← applied
component=summarize tokens.before=407 tokens.after=407 tokens.saved=0   reverted=true    ← declined
```

`reverted=true` means it ran but the output was not smaller, so the original was
kept — correct behaviour, but a sign that one of the three problems above
applies. Single-digit `duration_ms` alongside it means the model was never
called: check `base_url` and the triggers.

Two more consequences worth planning for:

- **It is slow.** Compaction runs in the background after the reply, so it never
  delays a request — but if the next turn arrives before the fold lands, there is
  no stored prefix to substitute and the saving silently doesn't happen. Allow
  more time between turns than the 3s above.
- **`summarize` reduces the message count**, so logs show `replaced=11 stored=3`.
  `toon` rewrites in place, so `stored` always equals `replaced` there.

## Using a hosted instance instead

If someone gives you a URL rather than asking you to run it, everything above
applies to the client side — same `/v1/chat/completions`, same `x-session-id`.
Two things differ:

- If the server uses a private CA (common on OpenShift), you need its CA file:
  `curl --cacert ca.pem ...`, or `export SSL_CERT_FILE=/path/ca.pem` for Python.
- If it has OIDC enabled, send `Authorization: Bearer <token>`.

Only `/v1/chat/completions` is meaningful behind an llm-d coordinator;
`/v1/responses` returns 404 there.

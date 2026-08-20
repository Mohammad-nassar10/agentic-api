# Trying the shared deployment

The stack is already running in `agentic-api-mtn` on `pokprod001`. You do **not**
need cluster access, `oc`, or a GPU — just an HTTP client.

If you want to run the whole thing yourself instead, see
[DEPLOY-DEMO.md](DEPLOY-DEMO.md). To run only agentic-api on a laptop against
your own backend, see [run-without-a-cluster.md](run-without-a-cluster.md).

---

## 1. The endpoint

```
https://agentic-api-agentic-api-mtn.apps.pokprod001.ete14.res.ibm.com/v1
model: Qwen/Qwen3-8B
```

It is OpenAI-compatible, so any OpenAI client works by changing `base_url`.

Two things to know before your first call:

- **Pass `-k` / `verify=False`.** The cluster uses a self-signed ingress
  certificate, so verification fails by default. Ask for the CA file if you would
  rather keep verification on.
- **The GPU workers are scaled to zero between demos.** Everything returns `502`
  until someone starts them — ask in the channel rather than debugging.

```bash
export API=https://agentic-api-agentic-api-mtn.apps.pokprod001.ete14.res.ibm.com
curl -sk -o /dev/null -w '%{http_code}\n' "$API/health"    # 200 = the gateway is up
```

`/health` answers even when the GPUs are down — it only tells you the gateway is
alive.

---

## 2. What it does

Chat Completions is stateless: your client resends the entire conversation every
turn, so the prompt grows without bound. This gateway remembers what it has
already seen for a session, stores a compacted stand-in, and substitutes it
before forwarding upstream.

You opt in with **one header**, `x-session-id`. Nothing else about your request
or the response changes.

The saving appears on the **second and later** turns — the first turn is what
gets stored.

---

## 3. Demo: see the saving

Send the same conversation twice with a session, then once without, and compare
`prompt_tokens`.

```bash
export API=https://agentic-api-agentic-api-mtn.apps.pokprod001.ete14.res.ibm.com
export SID="yourname-$(date +%s)"     # any stable string, unique to you
```

**Turn 1** — nothing stored yet:

```bash
cat > /tmp/t1.json <<'JSON'
{"model":"Qwen/Qwen3-8B","max_tokens":60,"messages":[
 {"role":"user","content":"Remind me how our deployment pipeline works, including the details we agreed last quarter."},
 {"role":"assistant","content":"Three stages: build, integration test, canary. Build compiles the workspace and tags an image with the commit sha. Integration tests run in an ephemeral namespace. Canary shifts five percent of traffic for thirty minutes while watching error rate and p99 latency."},
 {"role":"user","content":"And the database migration policy?"},
 {"role":"assistant","content":"Forward only, backwards compatible for one release. We never drop a column in the release that stops writing it. Migrations are numbered and the schema version is asserted at startup."},
 {"role":"user","content":"And incident response?"},
 {"role":"assistant","content":"Weekly on call rotation. Sev1 is customer facing errors above one percent and pages immediately. Sev2 is degraded latency, business hours only. Every incident gets a written retrospective within five working days."},
 {"role":"user","content":"Given all that, what should I check first if p99 latency doubles?"}
]}
JSON

curl -sk "$API/v1/chat/completions" -H 'content-type: application/json' \
  -H "x-session-id: $SID" -d @/tmp/t1.json | grep -o '"prompt_tokens":[0-9]*'
```

**Wait for the fold.** Compaction runs in the background *after* your reply, so
it never slows a request — but the next turn has to arrive after it lands:

```bash
sleep 15
```

**Turn 2** — same history plus two more messages:

```bash
python3 - <<'PY'
import json
d = json.load(open("/tmp/t1.json"))
d["messages"] += [
    {"role": "assistant", "content": "Start with KV cache utilisation and waiting queue depth on the decode pool."},
    {"role": "user", "content": "And if those look normal?"},
]
json.dump(d, open("/tmp/t2.json", "w"))
PY

# with the session — the stored prefix is substituted
curl -sk "$API/v1/chat/completions" -H 'content-type: application/json' \
  -H "x-session-id: $SID" -d @/tmp/t2.json | grep -o '"prompt_tokens":[0-9]*'

# control: identical body, no session header
curl -sk "$API/v1/chat/completions" -H 'content-type: application/json' \
  -d @/tmp/t2.json | grep -o '"prompt_tokens":[0-9]*'
```

Expected shape — the two numbers differ only because of the header:

```
turn 1            477
turn 2, session   215
turn 2, control   508     ← ~58% more prompt for the same conversation
```

Your exact numbers will vary with the compactor mode currently configured
(see §5).

---

## 3b. Demo: a scripted client (easier)

Hand-writing each turn gets tedious, and two turns barely shows the effect.
[`demo_agent.py`](demo_agent.py) behaves like a real client — it keeps the
conversation, appends each reply, and resends the whole history every turn. It
runs two identical conversations side by side, one with the session header and
one without, and prints both token counts.

Standard library only, no dependencies:

```bash
python3 demo_agent.py --turns 5
```

```
turn   with session    control    saved
----  -------------  ---------  -------
   1             21         21     0.0%
   2             77         77     0.0%
   3            135        135     0.0%
   4            179        193     7.3%
   5            215        249    13.7%
```

Both columns send byte-identical requests; the only difference is the
`x-session-id` header on the first.

The early turns match because the compactor has a minimum size below which it
does nothing — there is not yet enough history to be worth compacting. Once it
starts, the two columns diverge and keep diverging, which is the point: the
control grows without bound, the compacted one does not.

Useful flags:

| flag | why |
|---|---|
| `--turns 10` | longer run, wider gap |
| `--payload tool` | sends a 20-row JSON array in a `tool` message — use when the compactor is in `toon` mode |
| `--settle 20` | wait longer between turns if folds are not landing in time |
| `--api ...` | point at a different deployment |

## 4. Demo: the backend load headers

Every response carries the serving endpoint's live metrics, added by an llm-d
endpoint-picker plugin:

```bash
curl -sk -D - -o /dev/null "$API/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | grep -i '^x-llm-d-'
```

```
x-llm-d-kv-cache-utilization: 0.0000     fraction of KV cache in use
x-llm-d-waiting-queue: 0                 requests queued on that endpoint
x-llm-d-running-requests: 0              requests in flight
x-llm-d-metrics-age-ms: 141              how fresh the snapshot is
```

Zeros are normal on an idle pool. The age proves they are live readings rather
than defaults. The gateway can use these to compact only when the backend is
actually under pressure (§5).

---

## 5. Changing the configuration

### What you control as a client

| | how |
|---|---|
| enable compaction | send `x-session-id` |
| disable it | omit the header |
| separate conversations | use different session values |
| reset a session | use a new session value |

There is no way to start a session over under the same id — pick a new one.

### What needs cluster access

Ask whoever owns the namespace, or run these yourself if you have it. All are in
`agentic-api-mtn`.

**Compactor mode.** Two are useful:

| mode | what it does | effect |
|---|---|---|
| `toon` | re-encodes JSON arrays in `tool` messages as TOON | same message count, fewer tokens |
| `summarize` | an LLM compresses the middle of the transcript | **fewer messages** |

```bash
# switch modes — chosen by the container's args, NOT by a CONFIG env var
oc patch deploy/context-guru -n agentic-api-mtn --type=json -p \
  '[{"op":"replace","path":"/spec/template/spec/containers/0/args",
     "value":["--config","/app/configs/toon.yaml"]}]'
```

`toon` only rewrites JSON arrays inside `tool` messages, and needs ~10+ records
before it saves anything — so the §3 demo above shows nothing under `toon`. Use a
`tool` message holding a 20-element array instead.

**Compaction thresholds.** By default every turn is compacted. These limit it to
when the pool is busy, using the §4 metrics:

```bash
oc set env deploy/agentic-api -n agentic-api-mtn \
  COMPACTION_KV_CACHE_THRESHOLD=0.8 \
  COMPACTION_WAITING_QUEUE_THRESHOLD=10
```

Any one being met triggers compaction. Unset ones are ignored. Anything
uncertain — no thresholds, no metrics headers, stale readings — compacts as
before. A malformed value stops the server at startup rather than being silently
dropped.

Setting env vars restarts the pod, and the session database is in-container, so
**every configuration change wipes all stored sessions**. Your first turn
afterwards will show no saving.

**Starting the GPUs** (scale each separately — a combined `oc scale` silently
does nothing):

```bash
oc scale deploy/vllm-p --replicas=1 -n agentic-api-mtn
oc scale deploy/vllm-d --replicas=1 -n agentic-api-mtn
```

---

## 6. Troubleshooting

| symptom | cause |
|---|---|
| `curl: (60) SSL certificate problem` | add `-k`, or get the CA file |
| `502` on every request | GPU workers scaled to zero — ask in the channel |
| first request very slow | vLLM cold start; send a throwaway request first |
| no saving on turn 2 | the fold had not landed — wait longer between turns |
| still no saving | the payload does not suit the active compactor mode (§5) |
| `404` on `/v1/responses` | only `/v1/chat/completions` is served behind the coordinator |
| sessions disappeared | the gateway restarted; the database is in-container |

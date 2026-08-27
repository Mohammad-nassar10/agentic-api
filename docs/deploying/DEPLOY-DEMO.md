# Deploying the session-compaction demo

End-to-end instructions for standing up the full stack on a fresh Kubernetes or
OpenShift namespace.

Written to be followed literally, top to bottom. Every step has a verification
command and its expected output — **do not continue past a step whose check
fails**, because later components fail in confusing, indirect ways when an
earlier one is unhealthy.

---

## 1. What you are deploying

```
client
  │  POST /v1/chat/completions   + x-session-id
  ▼
agentic-api ───────────────► context-guru        (folds the session prefix,
  │  substitutes stored prefix                    off the request path)
  ▼
llm-d-coordinator  ────────► context-guru        (optional inline compaction,
  │  splits prefill / decode                      only with x-llm-d-optimization)
  ▼
istio gateway (HTTPRoute matches EPP-Phase header)
  ├── EPP prefill  ──► vllm-p   ┐
  └── EPP decode   ──► vllm-d   ┘ KV transfer over NIXL
```

Two independent compaction paths share one context-guru:

| | agentic-api | coordinator |
|---|---|---|
| what | session prefix across turns | one oversized request body |
| when | background, after the reply | synchronous, in-band |
| trigger | `x-session-id` present | `x-llm-d-optimization: compaction` + body > threshold |
| default | **on** | **off** (inert without the header) |

### Components

| component | image | port |
|---|---|---|
| agentic-api | `ghcr.io/mohammad-nassar10/agentic-api:session` | 9000 |
| llm-d-coordinator | `ghcr.io/ronenkat/llm-d-coordinator:dev` | 8080 |
| context-guru | `ghcr.io/ronenkat/context-guru-proxy` | 4000 |
| EPP (prefill) | `ghcr.io/llm-d/llm-d-inference-scheduler:latest` | 9002 |
| EPP (decode) | `ghcr.io/mohammad-nassar10/llm-d-router-endpoint-picker:metrics-headers` | 9002 |
| vLLM prefill/decode | `ghcr.io/llm-d/llm-d-cuda:latest` | 8000 |

---

## 2. Prerequisites

- A cluster with **2 free NVIDIA GPUs** (one prefill, one decode), each pod also
  requesting 8–16 CPU and 32–48Gi RAM.
- Gateway API + Istio installed, with an `istio` GatewayClass.
- The Gateway API Inference Extension CRDs, specifically
  `inferencepools.inference.networking.k8s.io` **v1**.
- `kubectl` (substitute for `oc` throughout) with admin rights in one namespace.

Verify before starting:

```bash
kubectl get crd inferencepools.inference.networking.k8s.io \
  -o jsonpath='{.spec.versions[*].name}{"\n"}'      # must include v1
kubectl get gatewayclass istio                       # must exist
kubectl get nodes -o json | jq '[.items[].status.allocatable["nvidia.com/gpu"] // "0" | tonumber] | add'
```

Set your namespace once; every later command uses it:

```bash
export NS=agentic-api-mtn
kubectl create namespace "$NS"
kubectl config set-context --current --namespace="$NS"
```

> **OpenShift:** vLLM needs to write into its image and use large shared memory.
> If your cluster enforces a restrictive SCC, grant the default service account
> `anyuid` before step 3:
> `oc adm policy add-scc-to-user anyuid -z default -n "$NS"`

Optional, only for gated models — Qwen3-8B is public, so you can skip this:

```bash
kubectl create secret generic hf-token --from-literal=token="$HF_TOKEN" -n "$NS"
```

---

## 3. vLLM prefill and decode

Both pods run the **same** command. They differ only in labels, which is how the
InferencePools tell them apart. `kv_role: kv_both` is correct for both — NIXL
negotiates direction per request.

> **Critical: `enableServiceLinks: false`.** Kubernetes injects `<SVCNAME>_PORT`
> env vars for every service in the namespace. A service named `vllm` produces
> `VLLM_PORT=tcp://10.x.x.x:8000`, which vLLM reads as its own `--port` and dies
> with `ValueError: ... appears to be a URI`. Disabling service links avoids this
> whatever your services are called. Do not remove it.

```bash
cat <<'YAML' | kubectl apply -n "$NS" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-p
spec:
  replicas: 1
  selector:
    matchLabels: {app: qwen3-pool, llm-d.ai/role: prefill}
  template:
    metadata:
      labels:
        app: qwen3-pool
        llm-d.ai/role: prefill
        llm-d.ai/component: prefill
    spec:
      enableServiceLinks: false
      containers:
      - name: vllm
        image: ghcr.io/llm-d/llm-d-cuda:latest
        command: ["vllm", "serve"]
        args:
        - Qwen/Qwen3-8B
        - --port=8000
        - --served-model-name=Qwen/Qwen3-8B
        - --disable-uvicorn-access-log
        - --gpu-memory-utilization=0.90
        - --block-size=128
        - --kv-transfer-config={"kv_connector":"NixlConnector","kv_role":"kv_both"}
        ports: [{containerPort: 8000}]
        env:
        - {name: HF_HOME, value: /model-cache}
        - {name: PYTHONHASHSEED, value: "42"}
        - name: HF_TOKEN
          valueFrom: {secretKeyRef: {name: hf-token, key: token, optional: true}}
        - name: VLLM_NIXL_SIDE_CHANNEL_HOST
          valueFrom: {fieldRef: {fieldPath: status.podIP}}
        - name: POD_IP
          valueFrom: {fieldRef: {fieldPath: status.podIP}}
        resources:
          requests: {cpu: "8", memory: 32Gi, nvidia.com/gpu: "1"}
          limits:   {cpu: "16", memory: 48Gi, nvidia.com/gpu: "1"}
        volumeMounts:
        - {name: shm, mountPath: /dev/shm}
        - {name: model-storage, mountPath: /model-cache}
      volumes:
      - {name: shm, emptyDir: {medium: Memory, sizeLimit: 10Gi}}
      - {name: model-storage, emptyDir: {sizeLimit: 20Gi}}
YAML
```

Create the decode deployment by swapping the three `prefill` values for `decode`:

```bash
kubectl get deploy vllm-p -n "$NS" -o yaml \
  | sed 's/vllm-p/vllm-d/; s/prefill/decode/g' \
  | kubectl apply -n "$NS" -f -
```

**Verify** — first start pulls a ~10GB image and downloads the model, so allow
10–15 minutes:

```bash
kubectl wait --for=condition=available deploy/vllm-p deploy/vllm-d -n "$NS" --timeout=20m
kubectl get pods -n "$NS" -l app=qwen3-pool
```

Expected: both `1/1 Running`. If a pod is `Pending`, you are out of GPU. If it is
`CrashLoopBackOff`, read the logs — the `VLLM_PORT` error above is the usual
cause and means `enableServiceLinks: false` is missing.

---

## 4. Endpoint pickers and InferencePools

Order matters: the pools reference the EPP **services** with `failureMode:
FailClose`, so create the EPPs first or all routing fails closed.

```bash
cat <<'YAML' | kubectl apply -n "$NS" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: epp-config-prefill
data:
  epp-config.yaml: |
    apiVersion: inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
    - type: queue-scorer
    - type: kv-cache-utilization-scorer
    - type: prefix-cache-scorer
    - type: metrics-data-source
      parameters:
        scheme: "http"
        path: "/metrics"
        insecureSkipVerify: true
    - type: core-metrics-extractor
    schedulingProfiles:
    - name: default
      plugins:
      - {pluginRef: queue-scorer, weight: 2}
      - {pluginRef: kv-cache-utilization-scorer, weight: 2}
      - {pluginRef: prefix-cache-scorer, weight: 3}
YAML
```

The decode config is the same **plus `metrics-to-headers`** (see step 8 for what
it does and why the extra flag is needed):

```bash
kubectl get cm epp-config-prefill -n "$NS" -o yaml \
  | sed 's/epp-config-prefill/epp-config-decode/; s/    - type: core-metrics-extractor/    - type: core-metrics-extractor\n    - type: metrics-to-headers/' \
  | kubectl apply -n "$NS" -f -
```

Now the EPP deployment and service, once per phase:

```bash
for PHASE in prefill decode; do
  IMAGE=ghcr.io/llm-d/llm-d-inference-scheduler:latest
  EXTRA=""
  if [ "$PHASE" = decode ]; then
    IMAGE=ghcr.io/mohammad-nassar10/llm-d-router-endpoint-picker:metrics-headers
    EXTRA='- --allow-experimental-plugins'
  fi
  cat <<YAML | kubectl apply -n "$NS" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: qwen3-epp-$PHASE
spec:
  replicas: 1
  selector: {matchLabels: {app: qwen3-epp-$PHASE}}
  template:
    metadata: {labels: {app: qwen3-epp-$PHASE}}
    spec:
      containers:
      - name: epp
        image: $IMAGE
        args:
        - --pool-name=qwen3-pool-$PHASE
        - --pool-namespace=$NS
        - --v=4
        - --zap-encoder=json
        - --grpc-port=9002
        - --grpc-health-port=9003
        - --config-file=/etc/epp/epp-config.yaml
        - --metrics-endpoint-auth=false
        - --secure-serving=false
        $EXTRA
        ports:
        - {containerPort: 9002}
        - {containerPort: 9003}
        - {containerPort: 9090, name: metrics}
        volumeMounts: [{name: config, mountPath: /etc/epp}]
      volumes:
      - {name: config, configMap: {name: epp-config-$PHASE}}
---
apiVersion: v1
kind: Service
metadata:
  name: qwen3-epp-$PHASE
spec:
  selector: {app: qwen3-epp-$PHASE}
  ports:
  - {name: grpc, port: 9002, targetPort: 9002}
  - {name: metrics, port: 9090, targetPort: 9090}
YAML
done
```

Then the pools, which bind each EPP to the pods carrying the matching role label:

```bash
for PHASE in prefill decode; do
  cat <<YAML | kubectl apply -n "$NS" -f -
apiVersion: inference.networking.k8s.io/v1
kind: InferencePool
metadata:
  name: qwen3-pool-$PHASE
spec:
  appProtocol: http
  selector:
    matchLabels: {llm-d.ai/role: $PHASE}
  targetPorts: [{number: 8000}]
  endpointPickerRef:
    group: ""
    kind: Service
    name: qwen3-epp-$PHASE
    port: {number: 9002}
    failureMode: FailClose
YAML
done
```

**Verify** — both EPPs must be `1/1` and free of errors:

```bash
kubectl get pods -n "$NS" -l 'app in (qwen3-epp-prefill,qwen3-epp-decode)'
kubectl logs deploy/qwen3-epp-decode -n "$NS" --tail=50 | grep -i 'error\|stability' || echo "clean"
```

Expected: `clean`. If you see `Plugin stability validation failed`, the
`--allow-experimental-plugins` flag did not make it onto the decode EPP.

---

## 5. Gateway and HTTPRoute

The route is what makes P/D disaggregation work: the coordinator sets an
`EPP-Phase` header, and each value routes to a different pool. Nothing else
distinguishes the two legs.

```bash
cat <<'YAML' | kubectl apply -n "$NS" -f -
apiVersion: gateway.networking.k8s.io/v1
kind: Gateway
metadata:
  name: inference-gateway
spec:
  gatewayClassName: istio
  listeners:
  - {name: http, port: 80, protocol: HTTP, allowedRoutes: {namespaces: {from: Same}}}
---
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: qwen3-pool-inference-route
spec:
  parentRefs: [{group: gateway.networking.k8s.io, kind: Gateway, name: inference-gateway}]
  rules:
  - matches:
    - path: {type: PathPrefix, value: /}
      headers: [{name: EPP-Phase, type: Exact, value: prefill}]
    backendRefs:
    - {group: inference.networking.k8s.io, kind: InferencePool, name: qwen3-pool-prefill, port: 8000, weight: 1}
    timeouts: {request: 120s}
  - matches:
    - path: {type: PathPrefix, value: /}
      headers: [{name: EPP-Phase, type: Exact, value: decode}]
    backendRefs:
    - {group: inference.networking.k8s.io, kind: InferencePool, name: qwen3-pool-decode, port: 8000, weight: 1}
    timeouts: {request: 120s}
YAML
```

**Verify:**

```bash
kubectl get gateway inference-gateway -n "$NS" \
  -o jsonpath='{.status.conditions[?(@.type=="Programmed")].status}{"\n"}'   # True
```

---

## 6. context-guru

The compactor. Three modes are available; the demo uses **toon**, which is
deterministic and needs no LLM of its own.

| mode | what it does | needs an LLM | changes message count |
|---|---|---|---|
| `toon` | re-encodes uniform JSON arrays in `tool` messages as TOON | no | no |
| `extract` | LLM writes a filter that deletes irrelevant lines | yes | no |
| `summarize` | LLM compresses the middle of a transcript | yes | **yes** |

> The image contains **no** configs and no `/app/configs` directory — supply them
> yourself. The mode comes from the container's `--config` flag; a `CONFIG`
> environment variable is ignored, because the image's own command line wins.

```bash
cat <<'YAML' | kubectl apply -n "$NS" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: context-guru-config
data:
  toon.yaml: |
    # Deterministic: re-encodes uniform JSON arrays in tool messages as TOON —
    # field names once, then one row per element. No LLM, nothing stored.
    pipeline: [format, toon]
    components:
      format:
        min_tokens: 50
      toon:
        min_tokens: 50
    store:
      enabled: false
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: context-guru
spec:
  replicas: 1
  selector: {matchLabels: {app: context-guru}}
  template:
    metadata: {labels: {app: context-guru}}
    spec:
      containers:
      - name: context-guru
        image: ghcr.io/ronenkat/context-guru-proxy
        # The mode comes from this flag. A CONFIG env var is ignored — the
        # image's own command line always wins.
        args: ["--config", "/app/configs/toon.yaml"]
        ports: [{containerPort: 4000, name: http}]
        volumeMounts: [{name: config, mountPath: /app/configs, readOnly: true}]
      volumes:
      - {name: config, configMap: {name: context-guru-config}}
---
apiVersion: v1
kind: Service
metadata:
  name: context-guru
spec:
  selector: {app: context-guru}
  ports: [{port: 4000, targetPort: 4000}]
YAML
```

**Verify** it folds an array on its own:

```bash
kubectl port-forward -n "$NS" svc/context-guru 4000:4000 >/dev/null 2>&1 &
sleep 3
curl -s localhost:4000/compact -H 'content-type: application/json' -d '{"model":"m","messages":[
 {"role":"tool","tool_call_id":"c1","content":"[{\"id\":1,\"name\":\"ana\"},{\"id\":2,\"name\":\"bo\"}]"}]}'
kill %1
```

Expected — field names once, then one row each:

```
[2]{id,name}: 1,ana 2,bo
```

> **Two limits worth knowing.** toon only rewrites JSON arrays inside `tool`
> messages; arrays in user messages pass through untouched. And below roughly ten
> records the TOON header costs more than it saves, so small payloads can come
> back *larger*. Use a 20+ record array in demos.

### Switching to an LLM-backed mode

Add the config to the ConfigMap and repoint the flag — the mode is chosen by the
container's `args`, not by an environment variable:

```bash
kubectl patch deploy/context-guru -n "$NS" --type=json -p \
  '[{"op":"replace","path":"/spec/template/spec/containers/0/args",
     "value":["--config","/app/configs/summarize.yaml"]}]'
```

A `summarize.yaml` to add under the ConfigMap's `data:`:

```yaml
  summarize.yaml: |
    pipeline: [summarize]
    components:
      summarize:
        summary_level: concise
        keep_last: 1
        min_tokens: 50
        resummarize_tokens: 6000
        marker_mode: "off"
        trigger:
          min_messages: 2            # defaults are 12 / 40000 — far too high
          min_request_tokens: 100    # for a demo-sized conversation
        model:
          source: config
          provider: openai
          base_url: http://llm-d-coordinator:8080   # NO trailing /v1
          model: Qwen/Qwen3-8B
          api_key: "dummy"
    store:
      enabled: false
```

### Two settings that silently produce nothing if wrong

**`base_url` must not end in `/v1`.** context-guru appends the full
`/v1/chat/completions` itself. With a trailing `/v1` the request goes to
`/v1/v1/chat/completions`, matches no route, and the compactor returns the input
unchanged in a few hundred milliseconds — no error anywhere. The tell is the
coordinator log staying silent: a working call adds `sending request` lines
there and takes seconds, not milliseconds.

**Reasoning models must have thinking disabled**, or summarization can never
help. Qwen3 emits a `<think>` block before its answer; asked to summarize a
52-token input it produced 400 completion tokens of reasoning and hit the cap.
The "summary" is then longer than the text it would replace, so context-guru
discards it — correctly, but it looks like a broken compactor. Set the default
server-side, **on both workers**, since a chat template that differs between
prefill and decode breaks P/D:

```bash
for D in vllm-p vllm-d; do
  kubectl patch deploy/$D -n "$NS" --type=json -p \
    "[{\"op\":\"add\",\"path\":\"/spec/template/spec/containers/0/args/-\",
       \"value\":\"--default-chat-template-kwargs={\\\"enable_thinking\\\": false}\"}]"
done
kubectl rollout status deploy/vllm-p -n "$NS" --timeout=900s
kubectl rollout status deploy/vllm-d -n "$NS" --timeout=900s
```

Measured effect on the same prompt: **400 completion tokens → 41**, and
compaction latency **14.3s → 1.1s**.

### Reading context-guru's log

Every call logs one line, and `reverted` is the field that matters:

```
component=summarize tokens.before=407 tokens.after=161 tokens.saved=246 reverted=false   ← applied
component=summarize tokens.before=407 tokens.after=407 tokens.saved=0   reverted=true    ← declined
```

`reverted=true` means it ran but the result was not smaller, so it kept the
original. That is correct behaviour, not a failure — but it means something
upstream needs fixing: usually thinking left enabled, or a payload too small for
a summary to win. A `duration_ms` in single digits with `reverted=true` means the
model was never called at all: check `base_url` and the triggers.

### Three things to weigh first

- **Point it at the coordinator, not agentic-api.** Nothing in that chain sets
  `x-llm-d-optimization`, so the coordinator's own inline-compaction step stays
  inert and there is no loop. The gateway cannot be used directly — its
  HTTPRoute only matches on `EPP-Phase`, so an unlabelled request 404s.
- **Lower the triggers or nothing fires.** The shipped defaults assume 20k–40k
  token payloads; a demo conversation is skipped silently and returned unchanged,
  which looks exactly like a broken deployment.
- **It competes with serving.** Compaction runs on the same model and GPUs as
  inference, and with compaction thresholds configured it fires precisely when
  the pool is already saturated. A small dedicated model avoids this; the
  shipped configs default to `gpt-4o-mini` for that reason.

Folding also becomes slow enough to miss its window: it runs after the reply, so
if the next turn arrives first there is no stored prefix to substitute and the
saving silently doesn't happen. Allow more time between turns than the demo's 3s.

### Measured

An 11-message conversation, via agentic-api with the summarize mode configured
as above:

```
folded session prefix      replaced=11 stored=3
substituted stored prefix  replaced=11 upstream_messages=5 client_messages=13
```

215 prompt tokens with the session header against 508 without — **58%**. Unlike
toon, `stored` is lower than `replaced`: the summarizer collapses messages rather
than only shrinking them.

---

## 7. llm-d-coordinator

The entry point. It sits **in front of** the gateway — it is not a sidecar and
not behind Envoy. Its pipeline runs compaction (inert by default), then prefill,
then decode, and reverse-proxies the decode response back to the caller.

```bash
cat <<YAML | kubectl apply -n "$NS" -f -
apiVersion: v1
kind: ConfigMap
metadata:
  name: llm-d-coordinator-config
data:
  coordinator.yaml: |
    log_level: 2
    server:
      listen_addr: ":8080"
      read_timeout: 30s
      write_timeout: 120s
      shutdown_timeout: 25s
    gateway:
      address: "http://inference-gateway-istio:80"
      max_idle_conns_per_host: 200
      idle_conn_timeout: 90s
      timeout: 60s
    pipeline:
      kv_connector: kv-nixl
      use_openai_format: true
      steps:
      - type: request-inline-compaction
        params:
          address: "http://context-guru.$NS.svc.cluster.local:4000/compact"
          timeout: "10s"
          max_idle_conns_per_host: 100
          idle_conn_timeout: 90s
          min_size_threshold: 1000
      - type: prefill
      - type: decode
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-d-coordinator
spec:
  replicas: 1
  selector: {matchLabels: {app: llm-d-coordinator}}
  template:
    metadata: {labels: {app: llm-d-coordinator}}
    spec:
      containers:
      - name: coordinator
        image: ghcr.io/ronenkat/llm-d-coordinator:dev
        args: ["--config", "/etc/coordinator/coordinator.yaml"]
        ports: [{containerPort: 8080, name: http}]
        volumeMounts: [{name: config, mountPath: /etc/coordinator}]
      volumes:
      - {name: config, configMap: {name: llm-d-coordinator-config}}
---
apiVersion: v1
kind: Service
metadata:
  name: llm-d-coordinator
spec:
  selector: {app: llm-d-coordinator}
  ports: [{port: 8080, targetPort: 8080}]
YAML
```

**Verify the whole inference path** — this is the most valuable check in the
document, because it exercises coordinator → gateway → EPP → prefill → decode:

```bash
kubectl port-forward -n "$NS" svc/llm-d-coordinator 8080:8080 >/dev/null 2>&1 &
sleep 4
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | head -c 200; echo
kill %1
```

Expected: a JSON body with `"object":"chat.completion"`. A 5xx here means the
problem is below the coordinator — recheck steps 3–5 before going further.

The coordinator speaks **only** Chat Completions; `/v1/responses` returns 404.

---

## 8. The metrics-to-headers plugin (already enabled in step 4)

The decode EPP image carries a plugin that stamps the serving endpoint's live
metrics onto every response:

| header | meaning |
|---|---|
| `x-llm-d-kv-cache-utilization` | KV cache in use, a fraction in `[0,1]` |
| `x-llm-d-waiting-queue` | requests queued on that endpoint |
| `x-llm-d-running-requests` | requests in flight |
| `x-llm-d-metrics-age-ms` | age of the snapshot |

Two constraints that are easy to get wrong:

- **It must run on the decode EPP.** The coordinator consumes the prefill
  response internally and never proxies it, so headers set by the prefill EPP are
  discarded.
- **It is registered Alpha**, so the EPP refuses to start without
  `--allow-experimental-plugins`. Without the flag you get
  `Plugin stability validation failed` and a CrashLoopBackOff.

**Verify** (workers must be running):

```bash
kubectl port-forward -n "$NS" svc/llm-d-coordinator 8080:8080 >/dev/null 2>&1 &
sleep 4
curl -s -D - -o /dev/null localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | grep -i '^x-llm-d-'
kill %1
```

Expected — four headers. Zeros are correct on an idle pool; the age proves the
values are live rather than defaults:

```
x-llm-d-kv-cache-utilization: 0.0000
x-llm-d-metrics-age-ms: 267
x-llm-d-running-requests: 0
x-llm-d-waiting-queue: 0
```

---

## 9. agentic-api

The client-facing gateway. It stores a compacted stand-in for the history it has
already seen and substitutes it on later turns.

```bash
cat <<YAML | kubectl apply -n "$NS" -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-api
spec:
  replicas: 1
  selector: {matchLabels: {app: agentic-api}}
  template:
    metadata: {labels: {app: agentic-api}}
    spec:
      containers:
      - name: agentic-api
        image: ghcr.io/mohammad-nassar10/agentic-api:session
        env:
        - {name: LLM_API_BASE,         value: "http://llm-d-coordinator:8080"}
        - {name: COMPACTION_ADDRESS,   value: "http://context-guru:4000/compact"}
        - {name: GATEWAY_HOST,         value: "0.0.0.0"}
        - {name: GATEWAY_PORT,         value: "9000"}
        - {name: DATABASE_URL,         value: "sqlite://./agentic_api.db"}
        - {name: OPENAI_API_KEY,       value: "dummy"}
        - {name: SKIP_LLM_READY_CHECK, value: "true"}
        - {name: RUST_LOG,             value: "agentic_server=debug,agentic_core=debug"}
        ports: [{containerPort: 9000, name: http}]
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-api
spec:
  selector: {app: agentic-api}
  ports: [{name: http, port: 9000, targetPort: 9000}]
YAML
```

Three env vars decide the behaviour:

- `LLM_API_BASE` — must point at the **coordinator**, not the gateway.
- `COMPACTION_ADDRESS` — unset disables folding entirely; the gateway still
  works, it just never compacts.
- `SKIP_LLM_READY_CHECK=true` — the coordinator has no `/v1/models`, so the
  startup probe would otherwise never pass.

> **The database is in-container SQLite.** Every pod restart loses all stored
> session prefixes. Fine for a demo; swap `DATABASE_URL` for Postgres and add a
> volume for anything longer-lived.

Expose it. On OpenShift:

```bash
oc expose svc/agentic-api -n "$NS"
oc patch route agentic-api -n "$NS" -p \
  '{"spec":{"tls":{"termination":"edge","insecureEdgeTerminationPolicy":"Redirect"}}}'
export ROUTE="https://$(oc get route agentic-api -n "$NS" -o jsonpath='{.spec.host}')"
```

On plain Kubernetes, port-forward instead:

```bash
kubectl port-forward -n "$NS" svc/agentic-api 9000:9000 >/dev/null 2>&1 &
export ROUTE="http://localhost:9000"
```

**Verify:**

```bash
curl -sk -o /dev/null -w '%{http_code}\n' "$ROUTE/health"     # 200
```

---

## 10. Smoke test

```bash
curl -sk "$ROUTE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}' \
  | head -c 200; echo
```

A `chat.completion` object means every hop works. You are done deploying.

---

## 11. Demo: the compaction saving

Turn 1 establishes a session; turn 2 reuses it; the control run sends the same
body without the session header.

```bash
cat > /tmp/turn1.json <<'JSON'
{"model":"Qwen/Qwen3-8B","max_tokens":40,"messages":[
 {"role":"tool","tool_call_id":"c1","content":"[{\"id\":1,\"name\":\"ana\",\"role\":\"admin\"},{\"id\":2,\"name\":\"bo\",\"role\":\"member\"},{\"id\":3,\"name\":\"cy\",\"role\":\"admin\"},{\"id\":4,\"name\":\"di\",\"role\":\"member\"},{\"id\":5,\"name\":\"ed\",\"role\":\"admin\"},{\"id\":6,\"name\":\"fi\",\"role\":\"member\"},{\"id\":7,\"name\":\"gu\",\"role\":\"admin\"},{\"id\":8,\"name\":\"ha\",\"role\":\"member\"},{\"id\":9,\"name\":\"io\",\"role\":\"admin\"},{\"id\":10,\"name\":\"jo\",\"role\":\"member\"},{\"id\":11,\"name\":\"ka\",\"role\":\"admin\"},{\"id\":12,\"name\":\"li\",\"role\":\"member\"},{\"id\":13,\"name\":\"mo\",\"role\":\"admin\"},{\"id\":14,\"name\":\"ne\",\"role\":\"member\"},{\"id\":15,\"name\":\"om\",\"role\":\"admin\"},{\"id\":16,\"name\":\"pa\",\"role\":\"member\"},{\"id\":17,\"name\":\"qi\",\"role\":\"admin\"},{\"id\":18,\"name\":\"ro\",\"role\":\"member\"},{\"id\":19,\"name\":\"sa\",\"role\":\"admin\"},{\"id\":20,\"name\":\"ti\",\"role\":\"member\"}]"},
 {"role":"user","content":"How many admins are in that list?"}
]}
JSON

sed 's/"How many admins are in that list?"}$/"How many admins are in that list?"},\n {"role":"assistant","content":"There are 10 admins."},\n {"role":"user","content":"Name the first two."}/' \
  /tmp/turn1.json > /tmp/turn2.json

SID="demo-$(date +%s)"
for f in turn1 turn2; do
  printf '%s with session: ' "$f"
  curl -sk "$ROUTE/v1/chat/completions" -H 'content-type: application/json' \
    -H "x-session-id: $SID" -d @/tmp/$f.json | grep -o '"prompt_tokens":[0-9]*'
  sleep 3   # the fold runs in the background; give it time before the next turn
done

printf 'turn2 control (no session): '
curl -sk "$ROUTE/v1/chat/completions" -H 'content-type: application/json' \
  -d @/tmp/turn2.json | grep -o '"prompt_tokens":[0-9]*'
```

Expected shape — roughly 27% fewer prompt tokens on turn 2 versus the control:

```
turn1 with session: "prompt_tokens":278
turn2 with session: "prompt_tokens":220
turn2 control (no session): "prompt_tokens":300
```

What the server did:

```bash
kubectl logs deploy/agentic-api -n "$NS" --tail=50 \
  | sed 's/\x1b\[[0-9;]*m//g' | grep -E "$SID|pool signals"
```

```
folded session prefix      session=demo-… replaced=N stored=N
substituted stored prefix  session=demo-… replaced=N upstream_messages=… client_messages=4
```

Two lines, in that order: `folded` means agentic-api called context-guru after
turn 1; `substituted` means the stored prefix replaced the leading messages on
turn 2. If `folded` is missing, `COMPACTION_ADDRESS` is wrong. If `substituted`
is missing, the fold had not finished before turn 2 arrived — increase the sleep.

`stored` equalling `replaced` is expected: toon rewrites the tool message's
content in place rather than removing messages. **The win is tokens, not message
count.**

> **Version note.** The exact `replaced=N` depends on the image. Builds from
> before the reply-exclusion fix fold the assistant reply too and report one more
> than the client sent; current builds fold only what the client sent. Likewise,
> the `llm-d pool signals` log line exists only in builds that include the
> `PoolSignals` reader — with the published `:session` image you will see the
> headers on the wire (step 8) but not in agentic-api's log.

The saving is also **constant, not compounding** — toon folds the array once, and
later prose turns accumulate uncompacted on top of it.

---

## 12. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| vLLM `ValueError: ... appears to be a URI` | service-link env var collides with vLLM config | `enableServiceLinks: false` |
| vLLM `Pending` | no free GPU | free one or lower replicas |
| 502/503 through the route | workers scaled to 0 or still loading | scale up, wait for `1/1` |
| EPP `Plugin stability validation failed` | Alpha plugin without the flag | add `--allow-experimental-plugins` |
| No `x-llm-d-*` headers | plugin on prefill EPP, or missing from decode config | it must be on **decode** |
| No token saving | `COMPACTION_ADDRESS` unset, or array not in a `tool` message | check both |
| Saving is *negative* | fewer than ~10 records | use a bigger array |
| Sessions vanish | agentic-api restarted; SQLite is in-container | expected; use Postgres to persist |
| Coordinator returns 404 | you called `/v1/responses` | it only serves `/v1/chat/completions` |
| First request very slow | vLLM cold start | send one throwaway request |

Component-by-component bisection, from the bottom up — the first one that fails
is where the problem is:

```bash
kubectl get pods -n "$NS"                                   # 1. everything Running?
kubectl logs deploy/qwen3-epp-decode -n "$NS" --tail=30     # 2. EPP healthy?
kubectl port-forward -n "$NS" svc/llm-d-coordinator 8080:8080 &   # 3. inference path
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'
curl -sk "$ROUTE/health"                                    # 4. agentic-api
```

---

## 13. Cost control and teardown

GPUs bill while idle. Scale the workers down between sessions — **separately**,
as a combined `scale` of two deployments silently does nothing:

```bash
kubectl scale deploy/vllm-p --replicas=0 -n "$NS"
kubectl scale deploy/vllm-d --replicas=0 -n "$NS"
```

Everything else is cheap and can stay up. To remove the whole demo:

```bash
kubectl delete namespace "$NS"
```

---

## 14. Building the images yourself

Three of the six images are built from source; the rest are upstream releases.
Do this if you are changing the code, or if you want to depend on your own
registry rather than someone else's personal builds.

```bash
export REG=ghcr.io/<you>       # your registry; must be public, or see the note below
```

The `make` targets pick a container tool themselves — `CONTAINER_RUNTIME`
prefers `docker` and falls back to `podman`. The hand-run commands below use
`docker`; substitute `podman` if that is what you have, or force the make
targets with `CONTAINER_RUNTIME=podman`.

> **Always pass the image variables on the `make` command line, not via
> `export`.** The makefiles declare them with `?=`, so an exported value competes
> with the file's own default and the winner depends on include order — the root
> `Makefile` and `Makefile.coord.mk` set *different* defaults for
> `IMAGE_REGISTRY`. A command-line assignment overrides both unconditionally.

### 14.1 EPP with the metrics-to-headers plugin

From the llm-d-router fork, branch `metrics-to-headers`:

```bash
git clone -b metrics-to-headers https://github.com/Mohammad-nassar10/llm-d-router.git
cd llm-d-router
make image-build-epp image-push-epp \
  EPP_IMAGE="$REG/llm-d-router-endpoint-picker:metrics-headers"
```

### 14.2 Coordinator

Same repository. The coordinator targets live in `Makefile.coord.mk`, which the
root `Makefile` does **not** include — you must pass it with `-f`. There is no
`image-push-coordinator` target, so push by hand:

```bash
make -f Makefile.coord.mk image-build-coordinator \
  COORDINATOR_IMAGE="$REG/llm-d-coordinator:dev"
docker push "$REG/llm-d-coordinator:dev"
```

This replaces `ghcr.io/ronenkat/llm-d-coordinator:dev` in step 7. That published
image is a personal build of this same repository, not a separate project, so
building it yourself removes the dependency on someone else's registry.

### 14.3 agentic-api

From the agentic-api fork, branch `feat/session-prefix-store`:

```bash
git clone -b feat/session-prefix-store https://github.com/Mohammad-nassar10/agentic-api.git
cd agentic-api
docker build -t "$REG/agentic-api:session" .
docker push     "$REG/agentic-api:session"
```

Then point step 9 at your image.

### 14.4 context-guru

Third-party (`ghcr.io/ronenkat/context-guru-proxy`) with **no tag**, so it
resolves to `:latest` and can change under you without warning. There is no
public build recipe here. Pin the digest you tested against:

```bash
docker pull ghcr.io/ronenkat/context-guru-proxy
docker inspect --format '{{index .RepoDigests 0}}' ghcr.io/ronenkat/context-guru-proxy
# use the ghcr.io/ronenkat/context-guru-proxy@sha256:... form in step 6
```

### 14.5 Pinning everything for a reproducible run

Mutable tags are the main reason a demo that worked last week stops working.
Once the stack is up and verified, capture the digests actually running and use
those in your manifests:

```bash
kubectl get pods -n "$NS" -o json \
  | jq -r '.items[].status.containerStatuses[]? | "\(.image)\t\(.imageID)"' \
  | sort -u
```

### 14.6 Registry access

Make each package public, or create a pull secret in the namespace:

```bash
kubectl create secret docker-registry ghcr \
  --docker-server=ghcr.io --docker-username="<you>" --docker-password="$GITHUB_TOKEN" -n "$NS"
kubectl patch serviceaccount default -n "$NS" \
  -p '{"imagePullSecrets":[{"name":"ghcr"}]}'
```

A private image shows up as `ImagePullBackOff` with `denied` in
`kubectl describe pod` — not as an authentication error on the deployment.

### 14.7 Before opening a PR against llm-d-router

Run the repo's own gate — format, lint, govulncheck, and DCO sign-off:

```bash
git remote add upstream https://github.com/llm-d/llm-d-router.git   # once
git fetch upstream main      # signed-commits-check diffs against upstream/main
make presubmit
```

> **Do not commit the `IMAGE_REGISTRY` / `EPP_TAG` edits** some people make to the
> root `Makefile` to build under their own account. Passing the variables on the
> command line as above keeps that convenience out of the diff.

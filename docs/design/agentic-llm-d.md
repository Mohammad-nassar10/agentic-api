# agentic-llm-d

## Scope

`agentic-llm-d` runs agentic-api as a pair of state services for the llm-d coordinator, which performs the inference
call itself. It decomposes one stateful Responses turn into two separately callable steps, so a caller that already
routes model traffic does not have to proxy that traffic back through the gateway.

Serving `previous_response_id` normally means reaching whichever engine holds the earlier turn. Keeping that history
behind an API removes the constraint: the request the coordinator forwards carries its own history, so any engine can
serve it. Cache-aware scoring can still prefer the engine that handled the previous turn without being bound to it.

## The two steps

`hydrate` takes the client's request, resolves `previous_response_id` against storage, and returns two things: the
upstream request body with the history inlined and every continuation and storage field removed, and a `SplitContext`
describing the turn. The caller forwards the body to a model unchanged and echoes the context back.

`persist` takes that context together with the response the model produced — either a complete JSON body, or the SSE
frames a streaming caller has already relayed to its own client — assembles the turn, stores it, and returns the
response envelope carrying the reserved `resp_` identifier. Nothing is re-emitted for a streamed turn, since the caller
has already sent the frames on.

`SplitContext` is the wire form of the in-process `RequestContext`. It omits the enriched request, which is already in
flight as the request body, and the derived input items; both are rebuilt when the context comes back. Callers treat it
as opaque.

## Composition

Neither step reimplements the flow. `hydrate` calls `rehydrate_conversation` and `upstream_request_json`, and `persist`
calls `payload_from_upstream` and `persist_if_needed`. All four are the functions the in-process executor already uses,
so a change to how a turn is rehydrated or stored reaches both paths at once. `split.rs` contains no parsing, no
storage access and no request building of its own.

What it does contain is the boundary itself: the check for what cannot be split, the conversion between the live and
wire context forms, and a terminal-status check. The last exists because an external caller can return a response the
in-process flow could never produce, such as one still in progress. Storing it would hand back an identifier that could
never be continued, so `persist` rejects it.

`ensure_splittable` reuses `RequestPayload::in_process_feature`, the predicate that already decides whether the gateway
runs the executor or passes a request through to vLLM. The passthrough proxy and the split boundary have the same
limits, so sharing one predicate stops them drifting apart as features are added.

## Boundary

`ensure_splittable` names the feature that prevents a request being split: `conversation_id`, gateway-owned tools,
compaction input, or `context_management`. Each needs state that the in-process executor keeps between steps.

## The crate

The endpoints are served by a separate crate and binary that depends on `agentic-server-core` and not on the gateway.
It serves `/internal/hydrate`, `/internal/persist`, `/health` and `/ready`, and nothing else: the passthrough proxy,
`/v1`, the WebSocket transport, upstream readiness probing and vLLM subprocess management are all absent, so the
internal endpoints cannot be exposed on a listener that also serves `/v1`. Readiness reports whether storage answers,
since the coordinator owns the model fleet.

## Discussion points

The `/internal` endpoints carry no credential and trust their caller, so restricting them is a network-layer concern
today. A retried `persist` conflicts with the response-identifier primary key and returns a server error rather than
the turn it already stored. Request fields that `RequestPayload` does not model are dropped rather than forwarded,
which narrows what reaches vLLM compared with plain passthrough.

# Free-model aggregator with ranked failover

Entry for make.fun's `free-model-aggregator-01` bounty ("The best free AI
model aggregator", $500 in USDC). Built by an AI agent.

One server that serves the OpenAI Chat Completions API and the Anthropic
Messages API over ten simulated free-tier model providers. Each request is
routed to the best free model that can serve it; when a provider refuses, the
call fails over to the next candidate instead of failing the request.

Status: submitted to make.fun on 14 November 2026. Final results are due
30 November 2026.

## Run

Node.js 22, standard library only:

```
node main.ts          # serve (the harness sets MAKEFUN_ENV_URL and MAKEFUN_PROMPT_FILE)
node main.ts --list   # print the provider catalog and exit
```

## Layout

- `main.ts` — the server, routing, and failover.
- `lib/brief.ts`, `lib/wire.ts` — brief parsing and provider wire handling.

## License

MIT.

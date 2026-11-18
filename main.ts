// One server, two client wires, ten simulated free-tier providers.
//
// The trial runs this as `node main.ts` with Node's standard library alone.
// It reads the brief, probes every provider's catalog route, builds one
// catalog, registers its port and catalog with the env, serves the client's
// twenty requests, and exits 0 when the env says the workload is done.
//
// Routing ranks every free model that meets a request's requirements by the
// quality table's rating, best first, skipping wires that cannot carry what
// the request asks for. A provider that refuses a call drops a tier for the
// requests after it, so the load spreads over providers instead of draining
// one budget, and a refused call is never the request's answer: the next
// candidate serves it. Every provider call is serialized, so the provider
// that serves one call at a time never sees a second call in flight.
//
// `node main.ts --list` prints the same catalog document the register body
// carries and exits before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesFor,
  isFree,
  isRecord,
  joinBase,
  meetsRequirements,
  probeProvider,
  ratingRowFor,
  readBrief,
  type Brief,
  type Catalog,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/brief.ts';
import {
  anthropicError,
  anthropicModelList,
  fromAnthropic,
  fromOpenAI,
  openAIError,
  openAIModelList,
  parseUpstream,
  renderAnthropicJson,
  renderAnthropicStream,
  renderOpenAIJson,
  renderOpenAIStream,
  toAnthropic,
  toGemini,
  toOpenAI,
  type ClientWire,
  type NormRequest,
  type ServedModel,
  type UpstreamResult,
} from './lib/wire.ts';

const PROTOCOL_VERSION = 1;
const POLL_INTERVAL_MS = 100;
const PROBE_TIMEOUT_MS = 5000;
const CALL_TIMEOUT_MS = 5000;
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

let brief: Brief = { providers: [], ratings: [], workload: [], error: null };
let probes: Probe[] = [];
let catalog: Catalog = { providers: [] };
/** Requests served so far: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Consecutive refused calls per provider: the tier a provider routes in. */
const consecutiveFailures = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

async function load(): Promise<void> {
  brief = readBrief(PROMPT_FILE);
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(probes, brief.ratings);
}

/** The free models across the providers that answered, deduplicated by id. */
function servedModels(): ServedModel[] {
  const seen = new Set<string>();
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    const sorted = [...probe.models].sort((a, b) => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
    for (const model of sorted) {
      if (seen.has(model.id)) continue;
      const row = ratingRowFor(brief.ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every (provider, model) that may serve this request, best first: providers
 * with no recent refusal first, then by the quality table's rating, then by
 * provider and model id so the order is the same on every pass. An
 * Anthropic-wire provider never serves a JSON-mode request: its wire cannot
 * carry the request's JSON mode, and the provider would echo it absent.
 */
function candidatesFor(requires: Requirement, jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingRowFor(brief.ratings, probe.provider.id, model.id);
      if (!isFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesFor(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  const tierOf = (c: Candidate): number => {
    const fails = consecutiveFailures.get(c.probe.provider.id) ?? 0;
    return fails === 0 ? 0 : fails === 1 ? 1 : 2;
  };
  out.sort((a, b) => {
    const tier = tierOf(a) - tierOf(b);
    if (tier !== 0) return tier;
    if (b.rating !== a.rating) return b.rating - a.rating;
    if (a.probe.provider.id !== b.probe.provider.id) return a.probe.provider.id < b.probe.provider.id ? -1 : 1;
    return a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0;
  });
  return out;
}

/** The requirements of the request now arriving: the manifest's row, else the body's own. */
function requirementsFor(req: NormRequest): Requirement {
  const entry = brief.workload[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return {
    tools: req.tools.length > 0,
    vision: req.turns.some((t) => t.images.length > 0),
    json: req.jsonMode,
  };
}

/** One in-flight provider call at most: a chain every call waits its turn on. */
let callQueue: Promise<void> = Promise.resolve();
function withCallSlot<T>(fn: () => Promise<T>): Promise<T> {
  const run = callQueue.then(fn, fn);
  callQueue = run.then(
    () => undefined,
    () => undefined,
  );
  return run;
}

type CallOutcome =
  | { kind: 'served'; result: UpstreamResult }
  | { kind: 'refused'; status: number }
  | { kind: 'unparseable' };

/** One inference call to one candidate. A 2xx that parses is served; anything else refuses. */
async function callCandidate(
  candidate: Candidate,
  req: NormRequest,
  anthropicVersion: string,
): Promise<CallOutcome> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  let url: string;
  let body: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    body = toOpenAI(req, candidate.model.id);
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = anthropicVersion;
    body = toAnthropic(req, candidate.model.id);
  } else {
    const verb = req.stream ? 'streamGenerateContent' : 'generateContent';
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${verb}`;
    body = toGemini(req);
  }
  let res: Response;
  try {
    res = await fetch(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(CALL_TIMEOUT_MS),
    });
  } catch (err) {
    log(`call to ${provider.id}/${candidate.model.id} failed: ${(err as Error).message}`);
    return { kind: 'refused', status: 0 };
  }
  const text = await res.text();
  if (!res.ok) {
    log(`call to ${provider.id}/${candidate.model.id} refused: HTTP ${res.status}`);
    return { kind: 'refused', status: res.status };
  }
  const result = parseUpstream(provider.wire, req.stream, text);
  if (result === null) {
    log(`call to ${provider.id}/${candidate.model.id} answered 200 with an unparseable body`);
    return { kind: 'unparseable' };
  }
  return { kind: 'served', result };
}

/**
 * Serve one client request: try the ranked candidates in order, then once
 * more from the top, and answer with the first provider that serves it. A
 * provider that refuses drops a tier for later requests; one that serves
 * clears its refusals. Stops at the first success: exactly one provider
 * call may answer a request.
 *
 * Each candidate gets two consecutive attempts before the next candidate is
 * tried: a provider's refusal may be one scheduled call number rather than
 * a spent budget, and the very next call number may serve, so retrying the
 * best model once recovers the request at its best rating instead of
 * dropping to a weaker model. A second refusal moves on.
 */
async function serve(req: NormRequest, anthropicVersion: string): Promise<{ result: UpstreamResult; model: string } | null> {
  const requires = requirementsFor(req);
  const ranked = candidatesFor(requires, req.jsonMode);
  if (ranked.length === 0) return null;
  return withCallSlot(async () => {
    for (let pass = 0; pass < 2; pass++) {
      for (const candidate of ranked) {
        for (let attempt = 0; attempt < 2; attempt++) {
          const outcome = await callCandidate(candidate, req, anthropicVersion);
          const providerId = candidate.probe.provider.id;
          if (outcome.kind === 'served') {
            consecutiveFailures.set(providerId, 0);
            return { result: outcome.result, model: candidate.model.id };
          }
          if (outcome.kind === 'unparseable') {
            // The provider recorded a success: no other provider may answer.
            return null;
          }
          consecutiveFailures.set(providerId, (consecutiveFailures.get(providerId) ?? 0) + 1);
        }
      }
    }
    return null;
  });
}

async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendStream(res: ServerResponse, text: string): void {
  res.writeHead(200, {
    'content-type': 'text/event-stream',
    'cache-control': 'no-cache',
    connection: 'keep-alive',
    'content-length': Buffer.byteLength(text),
  });
  res.end(text);
}

function sendError(res: ServerResponse, wire: ClientWire, status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

async function infer(req: IncomingMessage, res: ServerResponse, wire: ClientWire): Promise<void> {
  const body = await readJson(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }
  const norm = wire === 'openai' ? fromOpenAI(body) : fromAnthropic(body);
  const version = typeof req.headers['anthropic-version'] === 'string' ? req.headers['anthropic-version'] : '2023-06-01';
  const served = await serve(norm, version);
  if (served === null) {
    sendError(
      res,
      wire,
      503,
      wire === 'anthropic' ? 'overloaded_error' : 'no_free_model',
      'no free model on the directory served this request',
    );
    return;
  }
  if (norm.stream) {
    const text =
      wire === 'openai' ? renderOpenAIStream(served.result, served.model) : renderAnthropicStream(served.result, served.model);
    sendStream(res, text);
    return;
  }
  sendJson(res, 200, wire === 'openai' ? renderOpenAIJson(served.result, served.model) : renderAnthropicJson(served.result, served.model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await infer(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await infer(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

async function register(port: number, report: Catalog): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work and exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) {
        sendError(
          res,
          req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai',
          500,
          'internal_error',
          'the server failed to answer',
        );
      }
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

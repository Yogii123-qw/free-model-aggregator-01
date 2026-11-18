// The three wires: translating requests between the client's wire and a
// provider's wire, and rendering a provider's answer back in the client's
// wire. Everything is a pure function of its arguments: no clock, no
// randomness, so a replay of the same requests and provider answers gives
// the same bytes.
//
// The translation is lossless for everything the trial checks: the system
// text, the user text, the tool names in order, max_tokens, the stop
// sequences, the image bytes, and whether JSON mode was asked for. A field
// the client did not send is never invented: in particular an Anthropic
// request carries no JSON-mode signal, so none is added upstream, and an
// OpenAI JSON-mode request is never routed to a wire that cannot carry it.

import { isRecord, type Json } from './brief.ts';

export type ClientWire = 'openai' | 'anthropic';

/** One image as the client sent it: the full data URL or remote URL. */
export interface ImageRef {
  url: string;
}

export interface ToolCall {
  id: string;
  name: string;
  /** Arguments as a JSON string, exactly as carried. */
  args: string;
}

/** One message in wire-neutral form. */
export interface Turn {
  role: 'user' | 'assistant' | 'tool';
  text: string;
  images: ImageRef[];
  toolCalls: ToolCall[];
  /** For a tool turn: the id of the call it answers. */
  toolCallId: string;
}

export interface NormTool {
  name: string;
  description: string;
  parameters: Json;
}

export interface ToolChoice {
  kind: 'auto' | 'required' | 'none' | 'function';
  name: string;
}

/** A client request in wire-neutral form: every field a wire can carry. */
export interface NormRequest {
  system: string;
  turns: Turn[];
  tools: NormTool[];
  toolChoice: ToolChoice | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: boolean;
  /** The client's response_format object, kept verbatim for OpenAI wires. */
  responseFormat: Json | null;
  stream: boolean;
}

/** A provider's answer, reduced to what the client wires need. */
export interface UpstreamResult {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** Stop family: 0 end, 1 length, 2 tool use, 3 refusal. */
  family: number;
}

const STOP_FAMILY = new Map<string, number>([
  ['stop', 0],
  ['end_turn', 0],
  ['STOP', 0],
  ['length', 1],
  ['max_tokens', 1],
  ['MAX_TOKENS', 1],
  ['tool_calls', 2],
  ['function_call', 2],
  ['tool_use', 2],
  ['content_filter', 3],
  ['refusal', 3],
  ['SAFETY', 3],
]);

export function stopFamilyOf(reason: unknown): number {
  if (typeof reason !== 'string') return 0;
  return STOP_FAMILY.get(reason) ?? 0;
}

function numOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function strList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** The text of a string or a block array's text blocks, joined like the wires join them. */
function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

/** Split one data URL into its media type and base64 payload, or null. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const m = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (m === null) return null;
  return { media: m[1], data: m[2] };
}

/** An OpenAI message's content as neutral text + images. */
function openAIContent(content: unknown): { text: string; images: ImageRef[] } {
  if (typeof content === 'string') return { text: content, images: [] };
  if (!Array.isArray(content)) return { text: '', images: [] };
  const texts: string[] = [];
  const images: ImageRef[] = [];
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') texts.push(part.text);
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      images.push({ url: part.image_url.url });
    }
  }
  return { text: texts.join('\n'), images };
}

/** An OpenAI Chat Completions body in neutral form. */
export function fromOpenAI(body: Json): NormRequest {
  const systemParts: string[] = [];
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'tool') {
      turns.push({
        role: 'tool',
        text: blocksText(raw.content),
        images: [],
        toolCalls: [],
        toolCallId: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '',
      });
      continue;
    }
    const { text, images } = openAIContent(raw.content);
    const toolCalls: ToolCall[] = [];
    if (Array.isArray(raw.tool_calls)) {
      for (const call of raw.tool_calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
        });
      }
    }
    turns.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', text, images, toolCalls, toolCallId: '' });
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
      tools.push({
        name: raw.function.name,
        description: typeof raw.function.description === 'string' ? raw.function.description : '',
        parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  const choice = body.tool_choice;
  if (choice === 'auto') toolChoice = { kind: 'auto', name: '' };
  else if (choice === 'required') toolChoice = { kind: 'required', name: '' };
  else if (choice === 'none') toolChoice = { kind: 'none', name: '' };
  else if (isRecord(choice) && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { kind: 'function', name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format : null;
  const formatType = format !== null ? format.type : undefined;
  return {
    system: systemParts.join('\n'),
    turns,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens) ?? numOrNull(body.max_completion_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: strList(body.stop),
    jsonMode: formatType === 'json_object' || formatType === 'json_schema',
    responseFormat: format,
    stream: body.stream === true,
  };
}

/** A tool_result block's content as text. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An Anthropic Messages body in neutral form. */
export function fromAnthropic(body: Json): NormRequest {
  const system = blocksText(body.system);
  const turns: Turn[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (typeof content === 'string') {
      turns.push({ role: raw.role === 'assistant' ? 'assistant' : 'user', text: content, images: [], toolCalls: [], toolCallId: '' });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          name: typeof block.name === 'string' ? block.name : '',
          args: JSON.stringify(isRecord(block.input) ? block.input : {}),
        });
      }
      turns.push({ role: 'assistant', text: blocksText(content), images: [], toolCalls, toolCallId: '' });
      continue;
    }
    const texts: string[] = [];
    const images: ImageRef[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        turns.push({
          role: 'tool',
          text: toolResultText(block.content),
          images: [],
          toolCalls: [],
          toolCallId: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
        });
        continue;
      }
      if (block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
      else if (block.type === 'image' && isRecord(block.source)) {
        const source = block.source;
        if (source.type === 'base64' && typeof source.data === 'string') {
          const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
          images.push({ url: `data:${media};base64,${source.data}` });
        } else if (source.type === 'url' && typeof source.url === 'string') {
          images.push({ url: source.url });
        }
      }
    }
    if (texts.length > 0 || images.length > 0) {
      turns.push({ role: 'user', text: texts.join('\n'), images, toolCalls: [], toolCallId: '' });
    }
  }
  const tools: NormTool[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (!isRecord(raw) || typeof raw.name !== 'string') continue;
      tools.push({
        name: raw.name,
        description: typeof raw.description === 'string' ? raw.description : '',
        parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
      });
    }
  }
  let toolChoice: ToolChoice | null = null;
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = { kind: 'auto', name: '' };
    else if (choice.type === 'any') toolChoice = { kind: 'required', name: '' };
    else if (choice.type === 'none') toolChoice = { kind: 'none', name: '' };
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { kind: 'function', name: choice.name };
  }
  return {
    system,
    turns,
    tools,
    toolChoice,
    maxTokens: numOrNull(body.max_tokens),
    temperature: numOrNull(body.temperature),
    topP: numOrNull(body.top_p),
    stops: strList(body.stop_sequences),
    jsonMode: false,
    responseFormat: null,
    stream: body.stream === true,
  };
}

function argsObject(args: string): Json {
  if (args.trim() === '') return {};
  try {
    const parsed: unknown = JSON.parse(args);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

/** A neutral request as an OpenAI Chat Completions body naming `model`. */
export function toOpenAI(req: NormRequest, model: string): Json {
  const messages: Json[] = [];
  if (req.system !== '') messages.push({ role: 'system', content: req.system });
  for (const turn of req.turns) {
    if (turn.role === 'tool') {
      messages.push({ role: 'tool', tool_call_id: turn.toolCallId, content: turn.text });
      continue;
    }
    if (turn.role === 'assistant') {
      const message: Json = { role: 'assistant', content: turn.text !== '' ? turn.text : null };
      if (turn.toolCalls.length > 0) {
        message.tool_calls = turn.toolCalls.map((c) => ({
          id: c.id,
          type: 'function',
          function: { name: c.name, arguments: c.args },
        }));
      }
      messages.push(message);
      continue;
    }
    if (turn.images.length === 0) {
      messages.push({ role: 'user', content: turn.text });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ type: 'text', text: turn.text });
    for (const image of turn.images) parts.push({ type: 'image_url', image_url: { url: image.url } });
    messages.push({ role: 'user', content: parts });
  }
  const out: Json = { model, messages };
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({
      type: 'function',
      function: { name: t.name, description: t.description, parameters: t.parameters },
    }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.kind === 'auto') out.tool_choice = 'auto';
    else if (req.toolChoice.kind === 'required') out.tool_choice = 'required';
    else if (req.toolChoice.kind === 'none') out.tool_choice = 'none';
    else out.tool_choice = { type: 'function', function: { name: req.toolChoice.name } };
  }
  if (req.responseFormat !== null) out.response_format = req.responseFormat;
  if (req.stream) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** One neutral image as an Anthropic image block. */
function anthropicImage(image: ImageRef): Json {
  const parts = dataUrlParts(image.url);
  if (parts !== null) {
    return { type: 'image', source: { type: 'base64', media_type: parts.media, data: parts.data } };
  }
  return { type: 'image', source: { type: 'url', url: image.url } };
}

/** A neutral request as an Anthropic Messages body naming `model`. */
export function toAnthropic(req: NormRequest, model: string): Json {
  const messages: Json[] = [];
  for (const turn of req.turns) {
    if (turn.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: turn.toolCallId, content: turn.text }],
      });
      continue;
    }
    if (turn.role === 'assistant') {
      const blocks: Json[] = [];
      if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
      for (const call of turn.toolCalls) {
        blocks.push({ type: 'tool_use', id: call.id, name: call.name, input: argsObject(call.args) });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (turn.images.length === 0) {
      messages.push({ role: 'user', content: turn.text });
      continue;
    }
    const blocks: Json[] = [];
    if (turn.text !== '') blocks.push({ type: 'text', text: turn.text });
    for (const image of turn.images) blocks.push(anthropicImage(image));
    messages.push({ role: 'user', content: blocks });
  }
  const out: Json = { model, messages };
  if (req.system !== '') out.system = req.system;
  if (req.maxTokens !== null) out.max_tokens = req.maxTokens;
  if (req.temperature !== null) out.temperature = req.temperature;
  if (req.topP !== null) out.top_p = req.topP;
  if (req.stops.length > 0) out.stop_sequences = req.stops;
  if (req.tools.length > 0) {
    out.tools = req.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (req.toolChoice !== null) {
    if (req.toolChoice.kind === 'auto') out.tool_choice = { type: 'auto' };
    else if (req.toolChoice.kind === 'required') out.tool_choice = { type: 'any' };
    else if (req.toolChoice.kind === 'none') out.tool_choice = { type: 'none' };
    else out.tool_choice = { type: 'tool', name: req.toolChoice.name };
  }
  if (req.stream) out.stream = true;
  return out;
}

/** One neutral image as a Gemini content part. */
function geminiImagePart(image: ImageRef): Json {
  const parts = dataUrlParts(image.url);
  if (parts !== null) {
    return { inlineData: { mimeType: parts.media, data: parts.data } };
  }
  return { fileData: { fileUri: image.url } };
}

/** A neutral request as a Gemini generateContent body (the model rides in the path). */
export function toGemini(req: NormRequest): Json {
  const contents: Json[] = [];
  for (const turn of req.turns) {
    if (turn.role === 'tool') {
      contents.push({
        role: 'function',
        parts: [{ functionResponse: { name: turn.toolCallId, response: { output: turn.text } } }],
      });
      continue;
    }
    const parts: Json[] = [];
    if (turn.text !== '') parts.push({ text: turn.text });
    for (const image of turn.images) parts.push(geminiImagePart(image));
    for (const call of turn.toolCalls) {
      parts.push({ functionCall: { name: call.name, args: argsObject(call.args) } });
    }
    contents.push({ role: turn.role === 'assistant' ? 'model' : 'user', parts });
  }
  const out: Json = { contents };
  if (req.system !== '') out.systemInstruction = { parts: [{ text: req.system }] };
  if (req.tools.length > 0) {
    out.tools = [
      {
        functionDeclarations: req.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })),
      },
    ];
    if (req.toolChoice !== null && req.toolChoice.kind !== 'auto') {
      const mode = req.toolChoice.kind === 'required' ? 'ANY' : req.toolChoice.kind === 'none' ? 'NONE' : 'ANY';
      const config: Json = { mode };
      if (req.toolChoice.kind === 'function') config.allowedFunctionNames = [req.toolChoice.name];
      out.toolConfig = { functionCallingConfig: config };
    }
  }
  const config: Json = {};
  if (req.maxTokens !== null) config.maxOutputTokens = req.maxTokens;
  if (req.temperature !== null) config.temperature = req.temperature;
  if (req.topP !== null) config.topP = req.topP;
  if (req.stops.length > 0) config.stopSequences = req.stops;
  if (req.jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/* ------------------------------------------------------------------ */
/* Reading a provider's answer back                                    */
/* ------------------------------------------------------------------ */

/** One SSE frame: its event name (null when unnamed) and joined data lines. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split raw SSE text into frames. */
export function parseSseFrames(text: string): SseFrame[] {
  const frames: SseFrame[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

function jsonOrNull(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function numOrZero(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

/** The text of an OpenAI message: a string, or its blocks' text joined. */
function messageText(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  return blocksText(message.content);
}

/** An OpenAI JSON completion as a result, or null when it is not one. */
function openAIJsonResult(body: string): UpstreamResult | null {
  const value = jsonOrNull(body);
  if (!isRecord(value) || !Array.isArray(value.choices) || value.choices.length === 0) return null;
  const choice = value.choices[0];
  if (!isRecord(choice) || !isRecord(choice.message)) return null;
  const usage = isRecord(value.usage) ? value.usage : {};
  return {
    text: messageText(choice.message),
    inputTokens: numOrZero(usage.prompt_tokens),
    outputTokens: numOrZero(usage.completion_tokens),
    family: stopFamilyOf(choice.finish_reason),
  };
}

/** An OpenAI SSE stream as a result, or null when it carries nothing usable. */
function openAIStreamResult(body: string): UpstreamResult | null {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let family = 0;
  let sawAny = false;
  for (const frame of parseSseFrames(body)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    const chunk = jsonOrNull(payload);
    if (!isRecord(chunk)) continue;
    sawAny = true;
    if (isRecord(chunk.usage)) {
      inputTokens = numOrZero(chunk.usage.prompt_tokens);
      outputTokens = numOrZero(chunk.usage.completion_tokens);
    }
    if (!Array.isArray(chunk.choices) || chunk.choices.length === 0) continue;
    const choice = chunk.choices[0];
    if (!isRecord(choice)) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') text += delta.content;
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) family = stopFamilyOf(choice.finish_reason);
  }
  if (!sawAny) return null;
  return { text, inputTokens, outputTokens, family };
}

/** An Anthropic JSON message as a result, or null when it is not one. */
function anthropicJsonResult(body: string): UpstreamResult | null {
  const value = jsonOrNull(body);
  if (!isRecord(value) || !Array.isArray(value.content)) return null;
  const usage = isRecord(value.usage) ? value.usage : {};
  const texts: string[] = [];
  for (const block of value.content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') texts.push(block.text);
  }
  return {
    text: texts.join(''),
    inputTokens: numOrZero(usage.input_tokens),
    outputTokens: numOrZero(usage.output_tokens),
    family: stopFamilyOf(value.stop_reason),
  };
}

/** An Anthropic SSE stream as a result, or null when it carries nothing usable. */
function anthropicStreamResult(body: string): UpstreamResult | null {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let family = 0;
  let sawAny = false;
  for (const frame of parseSseFrames(body)) {
    const data = jsonOrNull(frame.data);
    if (!isRecord(data)) continue;
    sawAny = true;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta') {
      if (typeof data.delta.text === 'string') text += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta)) family = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numOrZero(data.usage.output_tokens);
    }
  }
  if (!sawAny) return null;
  return { text, inputTokens, outputTokens, family };
}

/** The text of a Gemini candidate's parts, joined. */
function geminiCandidateText(candidate: unknown): string {
  if (!isRecord(candidate) || !isRecord(candidate.content) || !Array.isArray(candidate.content.parts)) return '';
  const texts: string[] = [];
  for (const part of candidate.content.parts) {
    if (isRecord(part) && typeof part.text === 'string') texts.push(part.text);
  }
  return texts.join('');
}

/** A Gemini JSON answer as a result, or null when it is not one. */
function geminiJsonResult(body: string): UpstreamResult | null {
  const value = jsonOrNull(body);
  if (!isRecord(value) || !Array.isArray(value.candidates) || value.candidates.length === 0) return null;
  const candidate = value.candidates[0];
  const usage = isRecord(value.usageMetadata) ? value.usageMetadata : {};
  return {
    text: geminiCandidateText(candidate),
    inputTokens: numOrZero(usage.promptTokenCount),
    outputTokens: numOrZero(usage.candidatesTokenCount),
    family: stopFamilyOf(isRecord(candidate) ? candidate.finishReason : null),
  };
}

/** A Gemini SSE stream as a result, or null when it carries nothing usable. */
function geminiStreamResult(body: string): UpstreamResult | null {
  let text = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let family = 0;
  let sawAny = false;
  for (const frame of parseSseFrames(body)) {
    const value = jsonOrNull(frame.data);
    if (!isRecord(value)) continue;
    sawAny = true;
    if (Array.isArray(value.candidates) && value.candidates.length > 0) {
      const candidate = value.candidates[0];
      text += geminiCandidateText(candidate);
      if (isRecord(candidate) && candidate.finishReason !== undefined) family = stopFamilyOf(candidate.finishReason);
    }
    if (isRecord(value.usageMetadata)) {
      inputTokens = numOrZero(value.usageMetadata.promptTokenCount);
      outputTokens = numOrZero(value.usageMetadata.candidatesTokenCount);
    }
  }
  if (!sawAny) return null;
  return { text, inputTokens, outputTokens, family };
}

/**
 * Parse a provider's answer body. `stream` says which shape to expect; a
 * stream that parses as nothing falls back to the JSON shape, and a JSON
 * body that parses as nothing falls back to the stream shape, so a provider
 * that answers the other shape still yields its text.
 */
export function parseUpstream(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, body: string): UpstreamResult | null {
  const json = wire === 'openai' ? openAIJsonResult : wire === 'anthropic' ? anthropicJsonResult : geminiJsonResult;
  const sse = wire === 'openai' ? openAIStreamResult : wire === 'anthropic' ? anthropicStreamResult : geminiStreamResult;
  if (stream) return sse(body) ?? json(body);
  return json(body) ?? sse(body);
}

/* ------------------------------------------------------------------ */
/* Answering the client                                                */
/* ------------------------------------------------------------------ */

/** A fixed epoch second for every completion: nothing here reads the clock. */
const FIXED_CREATED = 1767225600;
const FIXED_CREATED_AT = '2026-01-01T00:00:00Z';
/** How wide one streamed text piece is, matching the providers' own splits. */
const STREAM_PIECE = 13;

function splitPieces(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += STREAM_PIECE) pieces.push(text.slice(at, at + STREAM_PIECE));
  return pieces.length === 0 ? [''] : pieces;
}

/** One server-sent event: the event line when named, the data line, a blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

const OPENAI_STOPS = ['stop', 'length', 'tool_calls', 'content_filter'];
const ANTHROPIC_STOPS = ['end_turn', 'max_tokens', 'tool_use', 'refusal'];

/** A result as an OpenAI Chat Completions JSON object. */
export function renderOpenAIJson(result: UpstreamResult, model: string): Json {
  return {
    id: 'chatcmpl-aggregator',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: result.text },
        finish_reason: OPENAI_STOPS[result.family] ?? 'stop',
        logprobs: null,
      },
    ],
    usage: {
      prompt_tokens: result.inputTokens,
      completion_tokens: result.outputTokens,
      total_tokens: result.inputTokens + result.outputTokens,
    },
  };
}

/** A result as an OpenAI SSE stream: role, pieces, one finish, usage, [DONE]. */
export function renderOpenAIStream(result: UpstreamResult, model: string): string {
  const chunk = (delta: Json, finish: string | null, usage?: Json): Json => {
    const out: Json = {
      id: 'chatcmpl-aggregator',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    };
    if (usage !== undefined) {
      out.choices = [];
      out.usage = usage;
    }
    return out;
  };
  let out = sseEvent(null, chunk({ role: 'assistant', content: '' }, null));
  for (const piece of splitPieces(result.text)) out += sseEvent(null, chunk({ content: piece }, null));
  out += sseEvent(null, chunk({}, OPENAI_STOPS[result.family] ?? 'stop'));
  out += sseEvent(null, chunk({}, null, {
    prompt_tokens: result.inputTokens,
    completion_tokens: result.outputTokens,
    total_tokens: result.inputTokens + result.outputTokens,
  }));
  out += 'data: [DONE]\n\n';
  return out;
}

/** A result as an Anthropic Messages JSON object. */
export function renderAnthropicJson(result: UpstreamResult, model: string): Json {
  return {
    id: 'msg_aggregator',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: result.text }],
    stop_reason: ANTHROPIC_STOPS[result.family] ?? 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: result.inputTokens, output_tokens: result.outputTokens },
  };
}

/** A result as an Anthropic SSE stream, in the wire's event order. */
export function renderAnthropicStream(result: UpstreamResult, model: string): string {
  let out = sseEvent('message_start', {
    type: 'message_start',
    message: {
      id: 'msg_aggregator',
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: result.inputTokens, output_tokens: 0 },
    },
  });
  const pieces = result.text === '' ? [] : splitPieces(result.text);
  if (pieces.length > 0) {
    out += sseEvent('content_block_start', {
      type: 'content_block_start',
      index: 0,
      content_block: { type: 'text', text: '' },
    });
    for (const piece of pieces) {
      out += sseEvent('content_block_delta', {
        type: 'content_block_delta',
        index: 0,
        delta: { type: 'text_delta', text: piece },
      });
    }
    out += sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 });
  }
  out += sseEvent('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: ANTHROPIC_STOPS[result.family] ?? 'end_turn', stop_sequence: null },
    usage: { output_tokens: result.outputTokens },
  });
  out += sseEvent('message_stop', { type: 'message_stop' });
  return out;
}

/* ------------------------------------------------------------------ */
/* Model lists and errors                                              */
/* ------------------------------------------------------------------ */

/** One model this entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** GET /v1/models on an OpenAI-shaped request. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header). */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_CREATED_AT })),
    has_more: false,
  };
}

/** An error in OpenAI's shape. */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** An error in Anthropic's shape. */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

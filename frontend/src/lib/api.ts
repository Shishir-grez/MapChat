import { createSSEParser } from './sse';
import type { Activity, Role, ServerHealth } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE_URL ?? '').replace(/\/+$/, '');

export class ApiError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly status = 0,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

/** A place as the server sends it (snake_case). */
export interface ServerPlace {
  index: number;
  label: string;
  name: string;
  address: string | null;
  lat: number;
  lon: number;
  approximate: boolean;
  maps_url: string;
}

export interface ChatDone {
  content: string;
  places: ServerPlace[];
  unresolved: string[];
}

export interface ChatRequestBody {
  message: string;
  history: { role: Role; content: string }[];
  context?: {
    map?: { center: [number, number]; zoom: number };
    user_location?: { lat: number; lon: number; accuracy?: number };
  };
}

export interface StreamHandlers {
  onStatus?: (activity: Activity) => void;
  onDelta?: (text: string) => void;
  onPlace?: (place: ServerPlace) => void;
  onDone?: (result: ChatDone) => void;
}

async function errorFrom(response: Response): Promise<ApiError> {
  let code = `http_${response.status}`;
  let message = `The server answered ${response.status} ${response.statusText}.`.replace(' .', '.');
  try {
    const body = await response.json();
    if (body?.error?.message) {
      message = String(body.error.message);
      code = String(body.error.code ?? code);
    }
  } catch {
    if (response.status === 502 || response.status === 503 || response.status === 504) {
      message = "Can't reach the MapChat server. Is the backend running?";
      code = 'unreachable';
    }
  }
  return new ApiError(message, code, response.status);
}

function networkError(error: unknown): unknown {
  if (error instanceof DOMException && error.name === 'AbortError') return error;
  if (error instanceof TypeError) {
    return new ApiError("Can't reach the MapChat server. Is the backend running?", 'unreachable');
  }
  return error;
}

export async function fetchHealth(signal?: AbortSignal): Promise<ServerHealth> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/health`, { signal, headers: { Accept: 'application/json' } });
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) throw await errorFrom(response);
  const contentType = response.headers.get('content-type') ?? '';
  if (!contentType.includes('json')) {
    throw new ApiError("Can't reach the MapChat server. Is the backend running?", 'unreachable');
  }
  return (await response.json()) as ServerHealth;
}

/**
 * POST /api/chat/stream and dispatch its events. Resolves after the `done` event;
 * rejects with ApiError on server/network errors and with an AbortError when aborted.
 */
export async function streamChat(
  body: ChatRequestBody,
  handlers: StreamHandlers,
  signal?: AbortSignal,
): Promise<ChatDone> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
      body: JSON.stringify(body),
      signal,
    });
  } catch (error) {
    throw networkError(error);
  }
  if (!response.ok) throw await errorFrom(response);
  if (!response.body) throw new ApiError('This browser cannot read streamed responses.', 'no_stream');

  let done: ChatDone | null = null;
  let failure: ApiError | null = null;

  const parser = createSSEParser(({ event, data }) => {
    let payload: unknown;
    try {
      payload = JSON.parse(data);
    } catch {
      return;
    }
    const p = payload as Record<string, unknown>;
    switch (event) {
      case 'status':
        handlers.onStatus?.({
          stage: (p.stage as Activity['stage']) ?? 'thinking',
          detail: typeof p.detail === 'string' ? p.detail : null,
        });
        break;
      case 'delta':
        if (typeof p.text === 'string') handlers.onDelta?.(p.text);
        break;
      case 'place':
        handlers.onPlace?.(p as unknown as ServerPlace);
        break;
      case 'done':
        done = p as unknown as ChatDone;
        handlers.onDone?.(done);
        break;
      case 'error':
        failure = new ApiError(String(p.message ?? 'The reply failed.'), String(p.code ?? 'error'));
        break;
    }
  });

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  try {
    for (;;) {
      const { value, done: finished } = await reader.read();
      if (finished) break;
      parser.push(value);
      if (failure) break;
    }
    parser.end();
  } catch (error) {
    throw networkError(error);
  } finally {
    reader.releaseLock();
  }

  if (failure) throw failure;
  if (!done) throw new ApiError('The connection closed before the reply finished.', 'incomplete');
  return done;
}

import type { Activity, Connection, ServerHealth } from '../lib/types';

/** Prompts that work well (and match the offline demo's canned answers). */
export const SUGGESTIONS: { title: string; prompt: string }[] = [
  { title: 'A slow day in Pune', prompt: 'Plan a slow day in Pune — history, good food and a quiet evening.' },
  { title: 'Hospitals for an emergency', prompt: 'Which are the best hospitals near Sassoon Road for an emergency?' },
  { title: 'Coffee to work from', prompt: 'Where can I get specialty coffee in Pune and work on my laptop?' },
  { title: 'A first weekend in Paris', prompt: 'Plan a first weekend in Paris, split by neighbourhood.' },
  { title: 'Monsoon treks', prompt: 'Suggest monsoon treks within two hours of Pune.' },
];

export function activityLabel(activity: Activity | null): string | null {
  if (!activity) return null;
  if (activity.detail) return activity.detail;
  switch (activity.stage) {
    case 'searching':
      return 'Searching the web';
    case 'locating':
      return 'Pinning places on the map';
    default:
      return 'Thinking';
  }
}

/** Short status for a header badge, or null when everything is normal. */
export function connectionLabel(connection: Connection, health: ServerHealth | null): string | null {
  switch (connection) {
    case 'offline':
      return 'Server offline';
    case 'unconfigured':
      return 'Model not configured';
    case 'checking':
      return null;
    default:
      return health?.demo ? 'Demo data' : null;
  }
}

/** Longer explanation to show when the app can't answer. */
export function connectionHelp(connection: Connection): string | null {
  if (connection === 'offline') {
    return "Can't reach the MapChat server. Start it with `python server.py` in backend/ — this page reconnects automatically.";
  }
  if (connection === 'unconfigured') {
    return 'The server is running without an OpenAI key. Add OPENAI_API_KEY to backend/.env, or start it with MAPCHAT_DEMO=1 to try the demo.';
  }
  return null;
}

/** Keyboard handling for a chat textarea: Enter sends, Shift+Enter adds a line, IME-safe. */
export function isSubmitKey(event: React.KeyboardEvent<HTMLTextAreaElement>): boolean {
  return event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing && event.keyCode !== 229;
}

/** Grow a textarea with its content up to `max` pixels. */
export function autosize(el: HTMLTextAreaElement | null, max = 200): void {
  if (!el) return;
  el.style.height = 'auto';
  el.style.height = `${Math.min(el.scrollHeight, max)}px`;
  el.style.overflowY = el.scrollHeight > max ? 'auto' : 'hidden';
}

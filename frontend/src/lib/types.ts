export type Role = 'user' | 'assistant';

/** A place mentioned in an assistant reply and found by the geocoder. */
export interface Place {
  /** Stable id: `${messageId}:${index}`. */
  id: string;
  /** 1-based number in reading order; matches the pin on the map. */
  index: number;
  /** The exact (normalized) text the model put inside the [LOCATION] tag. */
  label: string;
  /** Short name, e.g. "Shaniwar Wada". */
  name: string;
  /** Formatted address from the geocoder, if any. */
  address: string | null;
  lat: number;
  lon: number;
  /** True when the geocoder only matched part of the query. */
  approximate: boolean;
  /** Google Maps link for the place (no API key needed). */
  mapsUrl: string;
}

export type MessageStatus = 'streaming' | 'done' | 'error' | 'stopped';

export interface ChatMessage {
  id: string;
  role: Role;
  /** Raw text. Assistant replies keep their [LOCATION] tags; render with <MessageContent>. */
  content: string;
  createdAt: number;
  status: MessageStatus;
  places: Place[];
  /** Tagged labels the geocoder could not find. */
  unresolved: string[];
  error: string | null;
}

export type ActivityStage = 'thinking' | 'searching' | 'locating';

export interface Activity {
  stage: ActivityStage;
  /** Human-readable detail, e.g. 'Searching the web for “…”'. */
  detail: string | null;
}

export interface ServerHealth {
  status: 'ok';
  demo: boolean;
  llm: boolean;
  model: string | null;
  search: boolean;
  geocoder: 'google' | 'nominatim' | 'demo';
}

/** 'checking' until /api/health answers. */
export type Connection = 'checking' | 'online' | 'offline' | 'unconfigured';

export interface MapView {
  center: [number, number];
  zoom: number;
}

export type GeoStatus = 'off' | 'requesting' | 'on' | 'denied' | 'unavailable';

export interface UserLocation {
  lat: number;
  lon: number;
  accuracy: number;
}

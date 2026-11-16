import type { MapView } from './types';

/** Parse VITE_DEFAULT_VIEW ("lat,lon,zoom"); falls back to central Pune. */
export function parseDefaultView(raw: string | undefined): MapView {
  const fallback: MapView = { center: [18.5204, 73.8567], zoom: 12 }; // Pune
  if (!raw) return fallback;
  const [lat, lon, zoom] = raw.split(',').map((part) => Number(part.trim()));
  if ([lat, lon].some((n) => n === undefined || !Number.isFinite(n))) return fallback;
  return { center: [lat!, lon!], zoom: Number.isFinite(zoom) ? zoom! : fallback.zoom };
}

export const DEFAULT_VIEW = parseDefaultView(import.meta.env.VITE_DEFAULT_VIEW);

/**
 * Basemap presets. CARTO basemaps are free for non-commercial use and low volumes; check
 * https://carto.com/basemaps for the terms, or set VITE_TILE_URL to your own provider.
 * Attribution must stay visible (OpenStreetMap ODbL requirement).
 */

export interface TileConfig {
  url: string;
  attribution: string;
  subdomains?: string;
  maxZoom?: number;
}

const OSM = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';
const CARTO = `${OSM} &copy; <a href="https://carto.com/attributions">CARTO</a>`;

export const TILE_PRESETS = {
  /** Pale grey, minimal labels: the quietest backdrop. */
  positron: {
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attribution: CARTO,
    subdomains: 'abcd',
    maxZoom: 20,
  },
  /** Positron without labels (pair with a label overlay or use as pure texture). */
  positronNoLabels: {
    url: 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
    attribution: CARTO,
    subdomains: 'abcd',
    maxZoom: 20,
  },
  /** Soft colour: parks, water and roads in muted tones. */
  voyager: {
    url: 'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    attribution: CARTO,
    subdomains: 'abcd',
    maxZoom: 20,
  },
  /** Standard OpenStreetMap tiles (heavy use is not allowed by the OSMF policy). */
  osm: {
    url: 'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: OSM,
    maxZoom: 19,
  },
} satisfies Record<string, TileConfig>;

export type TilePreset = keyof typeof TILE_PRESETS;

/** The preset, unless VITE_TILE_URL overrides tiles for the whole app. */
export function resolveTiles(preset: TilePreset | TileConfig): TileConfig {
  const url = import.meta.env.VITE_TILE_URL;
  if (url) {
    return { url, attribution: import.meta.env.VITE_TILE_ATTRIBUTION || OSM, maxZoom: 19 };
  }
  return typeof preset === 'string' ? TILE_PRESETS[preset] : preset;
}

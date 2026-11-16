/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Absolute API origin when the frontend is not served behind the same origin (default: same origin). */
  readonly VITE_API_BASE_URL?: string;
  /** Dev-server proxy target for /api (default http://127.0.0.1:5000). */
  readonly VITE_API_PROXY_TARGET?: string;
  /** 'atlas' | 'meridian' | 'concierge' */
  readonly VITE_DEFAULT_VARIANT?: string;
  /** Set to 'off' to hide the UI-variant switcher. */
  readonly VITE_VARIANT_SWITCHER?: string;
  /** Override the map tiles: URL template and attribution HTML. */
  readonly VITE_TILE_URL?: string;
  readonly VITE_TILE_ATTRIBUTION?: string;
  /** Initial map view, "lat,lon,zoom" (default Pune: 18.5204,73.8567,12). */
  readonly VITE_DEFAULT_VIEW?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

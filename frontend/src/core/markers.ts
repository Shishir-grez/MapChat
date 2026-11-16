import type { Place } from '../lib/types';

/** The default pin: `.mc-pin > .mc-pin-index`. Numbers only — no user text in HTML. */
export const defaultRenderMarker = (place: Place): string =>
  `<div class="mc-pin"><span class="mc-pin-index">${place.index}</span></div>`;

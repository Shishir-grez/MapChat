import { createContext, useContext } from 'react';
import type { Activity, ChatMessage, Connection, GeoStatus, MapView, Place, ServerHealth, UserLocation } from '../lib/types';

/** Implemented by <PlaceMap>; the controller forwards `map.*` calls to it. */
export interface MapDriver {
  zoomIn(): void;
  zoomOut(): void;
  fitToPlaces(): void;
  flyTo(lat: number, lon: number, zoom?: number): void;
}

export interface MapApi extends MapDriver {
  /** Ask for the user's location (if needed) and fly to it. */
  locate(): void;
}

export interface GeoState {
  status: GeoStatus;
  location: UserLocation | null;
  error: string | null;
}

export interface MapChatController {
  // Conversation
  messages: ChatMessage[];
  isStreaming: boolean;
  /** What the assistant is doing before/around the text: thinking, searching, locating. */
  activity: Activity | null;
  send(text: string): void;
  stop(): void;
  /** Re-ask the last question when its reply failed or was stopped. */
  retry(): void;
  canRetry: boolean;
  /** Start a new conversation. */
  reset(): void;

  // Places on the map (they belong to the "active" assistant message)
  activeMessageId: string | null;
  activeMessage: ChatMessage | null;
  places: Place[];
  selectedPlace: Place | null;
  selectedPlaceId: string | null;
  hoveredPlaceId: string | null;
  /** Show this assistant message's places on the map. */
  focusMessage(messageId: string): void;
  /** Select a place by id (also activates its message); null clears. */
  selectPlace(placeId: string | null): void;
  hoverPlace(placeId: string | null): void;

  // Context sent with each question
  mapView: MapView;
  setMapView(view: MapView): void;
  /** Send the visible map area as context ("cafés around here"). */
  useMapArea: boolean;
  setUseMapArea(value: boolean): void;
  geo: GeoState;
  enableLocation(): void;
  disableLocation(): void;

  // Server
  connection: Connection;
  health: ServerHealth | null;
  recheckConnection(): void;

  // Map
  map: MapApi;
  registerMapDriver(driver: MapDriver | null): void;
}

export const MapChatContext = createContext<MapChatController | null>(null);

export function useMapChat(): MapChatController {
  const value = useContext(MapChatContext);
  if (!value) throw new Error('useMapChat must be used inside <MapChatProvider>');
  return value;
}


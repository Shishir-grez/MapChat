/**
 * The Leaflet map: numbered pins for the active reply, the user's location, and the
 * camera logic (fit new pins, fly to the selection, report the view back).
 *
 * Styling is up to each variant: pins render as `.mc-marker > .mc-pin` (override with
 * `renderMarker`) and expose their state as data attributes on `.mc-marker`:
 *   data-state="idle|hovered|selected"   data-dimmed="true|false"
 */
import { useEffect, useMemo, useRef, type ReactNode } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { AttributionControl, Circle, MapContainer, Marker, Popup, TileLayer, useMap } from 'react-leaflet';
import { resolveTiles, type TileConfig, type TilePreset } from '../lib/tiles';
import type { Place } from '../lib/types';
import { useMapChat, type MapDriver } from './context';
import { defaultRenderMarker } from './markers';

export interface MapPadding {
  top: number;
  right: number;
  bottom: number;
  left: number;
}

export interface PlaceMapProps {
  tiles?: TilePreset | TileConfig;
  className?: string;
  /** Pixels covered by floating UI on each side; pins are fitted inside the rest. */
  padding?: Partial<MapPadding>;
  /** Pin markup (HTML string). Only use numbers/static markup — never raw user text. */
  renderMarker?: (place: Place) => string;
  /** Pin box size and the point (from its top-left) that sits on the coordinate. */
  markerSize?: [number, number];
  markerAnchor?: [number, number];
  /** Popup body for the selected pin. Omit (or pass null) for no popups. */
  renderPopup?: ((place: Place) => ReactNode) | null;
  popupClassName?: string;
  attributionPosition?: L.ControlPosition;
  /** Max zoom when fitting several pins (default 15). */
  maxFitZoom?: number;
  /** Zoom used when flying to a single pin (default 15). */
  focusZoom?: number;
}

const DEFAULT_PADDING: MapPadding = { top: 0, right: 0, bottom: 0, left: 0 };
const EDGE = 56; // breathing room around fitted pins

function prefersReducedMotion(): boolean {
  return typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
}

// ------------------------------------------------------------------ camera + bridge

function MapBridge({ padding, maxFitZoom, focusZoom }: { padding: MapPadding; maxFitZoom: number; focusZoom: number }) {
  const map = useMap();
  const chat = useMapChat();
  const { places, activeMessageId, selectedPlace, registerMapDriver, setMapView } = chat;

  const paddingRef = useRef(padding);
  paddingRef.current = padding;
  const placesRef = useRef(places);
  placesRef.current = places;
  const userMovedRef = useRef(false);
  const lastActiveRef = useRef<string | null>(activeMessageId);
  const fitTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  const driver = useMemo<MapDriver>(() => {
    const animate = () => !prefersReducedMotion();

    const flyTo = (lat: number, lon: number, zoom?: number) => {
      const p = paddingRef.current;
      const z = zoom ?? Math.max(map.getZoom(), focusZoom);
      const offset = L.point((p.left - p.right) / 2, (p.top - p.bottom) / 2);
      const target = map.unproject(map.project([lat, lon], z).subtract(offset), z);
      if (animate()) map.flyTo(target, z, { duration: 0.85 });
      else map.setView(target, z, { animate: false });
    };

    const fitToPlaces = () => {
      const list = placesRef.current;
      if (list.length === 0) return;
      if (list.length === 1) {
        flyTo(list[0]!.lat, list[0]!.lon, Math.min(maxFitZoom, focusZoom));
        return;
      }
      const p = paddingRef.current;
      const bounds = L.latLngBounds(list.map((pl) => [pl.lat, pl.lon] as [number, number]));
      map.flyToBounds(bounds, {
        paddingTopLeft: [p.left + EDGE, p.top + EDGE],
        paddingBottomRight: [p.right + EDGE, p.bottom + EDGE],
        maxZoom: maxFitZoom,
        duration: 0.9,
        animate: animate(),
      });
    };

    return {
      zoomIn: () => map.zoomIn(1, { animate: animate() }),
      zoomOut: () => map.zoomOut(1, { animate: animate() }),
      fitToPlaces,
      flyTo,
    };
  }, [map, maxFitZoom, focusZoom]);

  useEffect(() => {
    registerMapDriver(driver);
    return () => registerMapDriver(null);
  }, [driver, registerMapDriver]);

  // Report the view; remember when the user moved the map themselves.
  useEffect(() => {
    const onMoveEnd = () => {
      const c = map.getCenter().wrap();
      setMapView({ center: [Number(c.lat.toFixed(5)), Number(c.lng.toFixed(5))], zoom: map.getZoom() });
    };
    const onUserMove = () => {
      userMovedRef.current = true;
    };
    const container = map.getContainer();
    map.on('moveend', onMoveEnd);
    map.on('dragstart', onUserMove);
    container.addEventListener('wheel', onUserMove, { passive: true });
    container.addEventListener('touchstart', onUserMove, { passive: true });
    return () => {
      map.off('moveend', onMoveEnd);
      map.off('dragstart', onUserMove);
      container.removeEventListener('wheel', onUserMove);
      container.removeEventListener('touchstart', onUserMove);
    };
  }, [map, setMapView]);

  // Keep Leaflet's idea of its size in sync with layout changes (panels opening, resizes).
  useEffect(() => {
    const container = map.getContainer();
    const observer = new ResizeObserver(() => map.invalidateSize({ debounceMoveend: true }));
    observer.observe(container);
    return () => observer.disconnect();
  }, [map]);

  // Fit pins: immediately when another reply becomes active, debounced while pins stream in.
  useEffect(() => {
    const switched = lastActiveRef.current !== activeMessageId;
    lastActiveRef.current = activeMessageId;
    if (switched) userMovedRef.current = false;
    if (places.length === 0) return;
    if (!switched && userMovedRef.current) return; // don't fight the user
    clearTimeout(fitTimer.current);
    fitTimer.current = setTimeout(() => driver.fitToPlaces(), switched ? 0 : 450);
    return () => clearTimeout(fitTimer.current);
  }, [activeMessageId, places.length, driver]);

  // Fly to the selected place.
  useEffect(() => {
    if (!selectedPlace) return;
    clearTimeout(fitTimer.current);
    driver.flyTo(selectedPlace.lat, selectedPlace.lon);
  }, [selectedPlace, driver]);

  return null;
}

// ------------------------------------------------------------------ markers

interface PlaceMarkerProps {
  place: Place;
  state: 'idle' | 'hovered' | 'selected';
  dimmed: boolean;
  icon: L.DivIcon;
  renderPopup: ((place: Place) => ReactNode) | null;
  popupClassName?: string;
}

function PlaceMarker({ place, state, dimmed, icon, renderPopup, popupClassName }: PlaceMarkerProps) {
  const { selectPlace, hoverPlace, selectedPlaceId } = useMapChat();
  const markerRef = useRef<L.Marker | null>(null);
  const selectedRef = useRef(selectedPlaceId);
  selectedRef.current = selectedPlaceId;

  useEffect(() => {
    const el = markerRef.current?.getElement();
    if (!el) return;
    el.dataset.state = state;
    el.dataset.dimmed = String(dimmed);
  });

  useEffect(() => {
    if (state === 'selected' && renderPopup) {
      // Let the fly-to start first so the popup doesn't push the map around.
      const t = setTimeout(() => markerRef.current?.openPopup(), 120);
      return () => clearTimeout(t);
    }
  }, [state, renderPopup]);

  const handlers = useMemo<L.LeafletEventHandlerFnMap>(
    () => ({
      click: () => selectPlace(place.id),
      keypress: (e) => {
        const key = (e as L.LeafletKeyboardEvent).originalEvent?.key;
        if (key === 'Enter' || key === ' ') selectPlace(place.id);
      },
      mouseover: () => hoverPlace(place.id),
      mouseout: () => hoverPlace(null),
      popupclose: () => {
        if (selectedRef.current === place.id) selectPlace(null);
      },
    }),
    [place.id, selectPlace, hoverPlace],
  );

  return (
    <Marker
      ref={markerRef}
      position={[place.lat, place.lon]}
      icon={icon}
      title={`${place.index}. ${place.name}`}
      alt={place.name}
      keyboard
      zIndexOffset={state === 'selected' ? 1000 : state === 'hovered' ? 500 : 0}
      eventHandlers={handlers}
    >
      {renderPopup ? (
        <Popup className={popupClassName} closeButton={false} autoPan={false} offset={[0, 0]}>
          {renderPopup(place)}
        </Popup>
      ) : null}
    </Marker>
  );
}

function UserLocationLayer() {
  const { geo } = useMapChat();
  const icon = useMemo(
    () =>
      L.divIcon({
        className: 'mc-user',
        html: '<span class="mc-user-halo"></span><span class="mc-user-dot"></span>',
        iconSize: [18, 18],
        iconAnchor: [9, 9],
      }),
    [],
  );
  if (geo.status !== 'on' || !geo.location) return null;
  const { lat, lon, accuracy } = geo.location;
  return (
    <>
      {accuracy > 25 && accuracy < 5000 ? (
        <Circle
          center={[lat, lon]}
          radius={accuracy}
          interactive={false}
          pathOptions={{ className: 'mc-user-accuracy', weight: 1, fillOpacity: 0.08, opacity: 0.35 }}
        />
      ) : null}
      <Marker position={[lat, lon]} icon={icon} interactive={false} keyboard={false} zIndexOffset={-100} />
    </>
  );
}

// ------------------------------------------------------------------ map

export function PlaceMap({
  tiles = 'positron',
  className,
  padding,
  renderMarker = defaultRenderMarker,
  markerSize = [30, 30],
  markerAnchor,
  renderPopup = null,
  popupClassName,
  attributionPosition = 'bottomright',
  maxFitZoom = 15,
  focusZoom = 15,
}: PlaceMapProps) {
  const { places, selectedPlaceId, hoveredPlaceId, mapView } = useMapChat();
  const initialView = useRef(mapView).current;
  const tile = resolveTiles(tiles);
  const pad = useMemo<MapPadding>(
    () => ({ ...DEFAULT_PADDING, ...padding }),
    [padding?.top, padding?.right, padding?.bottom, padding?.left], // eslint-disable-line react-hooks/exhaustive-deps
  );
  const anchor = markerAnchor ?? [markerSize[0] / 2, markerSize[1] / 2];

  const icons = useMemo(() => {
    const byId = new Map<string, L.DivIcon>();
    for (const place of places) {
      byId.set(
        place.id,
        L.divIcon({
          className: 'mc-marker',
          html: renderMarker(place),
          iconSize: markerSize,
          iconAnchor: anchor,
          popupAnchor: [0, -anchor[1]],
        }),
      );
    }
    return byId;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [places, renderMarker, markerSize[0], markerSize[1], anchor[0], anchor[1]]);

  const focusId = selectedPlaceId ?? hoveredPlaceId;

  return (
    <div className={['mc-map', className].filter(Boolean).join(' ')}>
      <MapContainer
        center={initialView.center}
        zoom={initialView.zoom}
        minZoom={2}
        zoomControl={false}
        attributionControl={false}
        worldCopyJump
        className="h-full w-full"
      >
        <TileLayer url={tile.url} attribution={tile.attribution} subdomains={tile.subdomains ?? 'abc'} maxZoom={tile.maxZoom ?? 19} />
        <AttributionControl position={attributionPosition} prefix={false} />
        <MapBridge padding={pad} maxFitZoom={maxFitZoom} focusZoom={focusZoom} />
        <UserLocationLayer />
        {places.map((place) => {
          const icon = icons.get(place.id);
          if (!icon) return null;
          const state = place.id === selectedPlaceId ? 'selected' : place.id === hoveredPlaceId ? 'hovered' : 'idle';
          return (
            <PlaceMarker
              key={place.id}
              place={place}
              icon={icon}
              state={state}
              dimmed={focusId !== null && focusId !== place.id}
              renderPopup={renderPopup}
              popupClassName={popupClassName}
            />
          );
        })}
      </MapContainer>
    </div>
  );
}

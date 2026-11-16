/**
 * MapChat's state and behaviour, shared by every UI variant.
 *
 * Variants render <PlaceMap>, <MessageContent> and their own chrome, and read/drive
 * everything else through `useMapChat()` (see ./context.ts for the full API).
 */
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';
import { ApiError, fetchHealth, streamChat, type ChatRequestBody, type ServerPlace } from '../lib/api';
import { readJSON, removeKey, writeJSON } from '../lib/storage';
import { DEFAULT_VIEW } from '../lib/view';
import { MapChatContext, type GeoState, type MapApi, type MapChatController, type MapDriver } from './context';
import type {
  Activity,
  ChatMessage,
  Connection,
  MapView,
  Place,
  ServerHealth,
} from '../lib/types';

// ------------------------------------------------------------------ helpers

const MAX_STORED_MESSAGES = 60;
const MAX_HISTORY = 12;
const MAX_HISTORY_CHARS = 6000;



function uid(): string {
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2, 10)}`;
}

function toPlace(messageId: string, p: ServerPlace): Place {
  return {
    id: `${messageId}:${p.index}`,
    index: p.index,
    label: p.label,
    name: p.name,
    address: p.address ?? null,
    lat: p.lat,
    lon: p.lon,
    approximate: Boolean(p.approximate),
    mapsUrl: p.maps_url,
  };
}

function isAbort(error: unknown): boolean {
  return error instanceof DOMException && error.name === 'AbortError';
}

function buildHistory(messages: ChatMessage[]): ChatRequestBody['history'] {
  return messages
    .filter((m) => m.content.trim() && (m.role === 'user' || m.status === 'done' || m.status === 'stopped'))
    .slice(-MAX_HISTORY)
    .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_HISTORY_CHARS) }));
}

interface StoredConversation {
  messages: ChatMessage[];
  activeMessageId: string | null;
}

function loadConversation(): StoredConversation {
  const stored = readJSON<StoredConversation | null>('conversation', null);
  if (!stored || !Array.isArray(stored.messages)) return { messages: [], activeMessageId: null };
  const messages = stored.messages
    .filter((m) => m && typeof m.id === 'string' && typeof m.content === 'string')
    .map((m) => ({
      ...m,
      places: Array.isArray(m.places) ? m.places : [],
      unresolved: Array.isArray(m.unresolved) ? m.unresolved : [],
      error: m.error ?? null,
      // A reply that was streaming when the page closed can't be resumed.
      status: m.status === 'streaming' ? ('stopped' as const) : m.status,
    }));
  const activeMessageId = messages.some((m) => m.id === stored.activeMessageId)
    ? stored.activeMessageId
    : ([...messages].reverse().find((m) => m.role === 'assistant')?.id ?? null);
  return { messages, activeMessageId };
}

interface Prefs {
  useMapArea: boolean;
  shareLocation: boolean;
}

// ------------------------------------------------------------------ provider

export function MapChatProvider({ children }: { children: ReactNode }) {
  const initial = useMemo(loadConversation, []);
  const initialPrefs = useMemo(() => readJSON<Prefs>('prefs', { useMapArea: true, shareLocation: false }), []);

  const [messages, setMessages] = useState<ChatMessage[]>(initial.messages);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(initial.activeMessageId);
  const [selectedPlaceId, setSelectedPlaceId] = useState<string | null>(null);
  const [hoveredPlaceId, setHoveredPlaceId] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activity, setActivity] = useState<Activity | null>(null);
  const [mapView, setMapViewState] = useState<MapView>(() => readJSON<MapView>('view', DEFAULT_VIEW));
  const [useMapArea, setUseMapAreaState] = useState(initialPrefs.useMapArea);
  const [geo, setGeo] = useState<GeoState>({ status: 'off', location: null, error: null });
  const [shareLocation, setShareLocation] = useState(initialPrefs.shareLocation);
  const [health, setHealth] = useState<ServerHealth | null>(null);
  const [connection, setConnection] = useState<Connection>('checking');

  const messagesRef = useRef(messages);
  messagesRef.current = messages;
  const abortRef = useRef<AbortController | null>(null);
  const streamingRef = useRef(false);
  const driverRef = useRef<MapDriver | null>(null);
  const flyToFixRef = useRef(false);
  const contextRef = useRef({ mapView, useMapArea, geo });
  contextRef.current = { mapView, useMapArea, geo };

  // ---------------------------------------------------------------- persistence

  useEffect(() => {
    if (isStreaming) return; // save once the reply settles
    if (messages.length === 0) {
      removeKey('conversation');
      return;
    }
    writeJSON('conversation', {
      messages: messages.slice(-MAX_STORED_MESSAGES),
      activeMessageId,
    } satisfies StoredConversation);
  }, [messages, activeMessageId, isStreaming]);

  useEffect(() => {
    writeJSON('prefs', { useMapArea, shareLocation } satisfies Prefs);
  }, [useMapArea, shareLocation]);

  // ---------------------------------------------------------------- server health

  const checkHealth = useCallback(async () => {
    try {
      const result = await fetchHealth();
      setHealth(result);
      setConnection(result.llm ? 'online' : 'unconfigured');
      return true;
    } catch {
      setConnection('offline');
      return false;
    }
  }, []);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      const ok = await checkHealth();
      // Keep retrying while the backend is down (e.g. started after the frontend).
      if (!cancelled && !ok) timer = setTimeout(poll, 8000);
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [checkHealth]);

  // ---------------------------------------------------------------- geolocation

  const enableLocation = useCallback(() => {
    if (!('geolocation' in navigator)) {
      setGeo({ status: 'unavailable', location: null, error: 'This browser cannot share its location.' });
      return;
    }
    setGeo((g) => ({ ...g, status: 'requesting', error: null }));
    navigator.geolocation.getCurrentPosition(
      (position) => {
        setGeo({
          status: 'on',
          location: {
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            accuracy: position.coords.accuracy,
          },
          error: null,
        });
        setShareLocation(true);
      },
      (error) => {
        const denied = error.code === error.PERMISSION_DENIED;
        setGeo({
          status: denied ? 'denied' : 'unavailable',
          location: null,
          error: denied ? 'Location access was blocked in the browser.' : 'Your location is unavailable right now.',
        });
        flyToFixRef.current = false;
        if (denied) setShareLocation(false);
      },
      { enableHighAccuracy: false, timeout: 15000, maximumAge: 5 * 60 * 1000 },
    );
  }, []);

  const disableLocation = useCallback(() => {
    setGeo({ status: 'off', location: null, error: null });
    setShareLocation(false);
    flyToFixRef.current = false;
  }, []);

  // Restore a previously shared location silently when permission is already granted.
  useEffect(() => {
    if (!initialPrefs.shareLocation || !navigator.permissions?.query) return;
    navigator.permissions
      .query({ name: 'geolocation' })
      .then((status) => {
        if (status.state === 'granted') enableLocation();
      })
      .catch(() => undefined);
  }, [enableLocation, initialPrefs.shareLocation]);

  useEffect(() => {
    if (flyToFixRef.current && geo.location) {
      flyToFixRef.current = false;
      driverRef.current?.flyTo(geo.location.lat, geo.location.lon, 15);
    }
  }, [geo.location]);

  // ---------------------------------------------------------------- chat

  const patchMessage = useCallback((id: string, update: (m: ChatMessage) => ChatMessage) => {
    setMessages((prev) => prev.map((m) => (m.id === id ? update(m) : m)));
  }, []);

  const runReply = useCallback(
    async (replyId: string, question: string, history: ChatRequestBody['history']) => {
      const controller = new AbortController();
      abortRef.current = controller;
      streamingRef.current = true;
      setIsStreaming(true);
      setActivity({ stage: 'thinking', detail: null });

      const { mapView: view, useMapArea: sendArea, geo: geoState } = contextRef.current;
      const context: ChatRequestBody['context'] = {};
      if (sendArea) context.map = { center: view.center, zoom: view.zoom };
      if (geoState.status === 'on' && geoState.location) {
        context.user_location = {
          lat: geoState.location.lat,
          lon: geoState.location.lon,
          accuracy: Math.round(geoState.location.accuracy),
        };
      }

      // Batch streamed text into one state update per animation frame.
      let pending = '';
      let frame = 0;
      let sawText = false;
      const flush = () => {
        frame = 0;
        if (!pending) return;
        const chunk = pending;
        pending = '';
        patchMessage(replyId, (m) => ({ ...m, content: m.content + chunk }));
      };
      const flushNow = () => {
        if (frame) cancelAnimationFrame(frame);
        flush();
      };

      try {
        const result = await streamChat(
          { message: question, history, context },
          {
            onStatus: (next) => setActivity(next),
            onDelta: (text) => {
              pending += text;
              if (!frame) frame = requestAnimationFrame(flush);
              if (!sawText) {
                sawText = true;
                setActivity(null);
              }
            },
            onPlace: (place) => {
              patchMessage(replyId, (m) =>
                m.places.some((p) => p.index === place.index)
                  ? m
                  : { ...m, places: [...m.places, toPlace(replyId, place)] },
              );
            },
          },
          controller.signal,
        );
        flushNow();
        patchMessage(replyId, (m) => ({
          ...m,
          content: result.content,
          places: result.places.map((p) => toPlace(replyId, p)),
          unresolved: result.unresolved ?? [],
          status: 'done',
          error: null,
        }));
        setConnection((c) => (c === 'offline' ? 'online' : c));
      } catch (error) {
        flushNow();
        if (isAbort(error)) {
          patchMessage(replyId, (m) => ({ ...m, status: 'stopped' }));
        } else {
          const message = error instanceof ApiError ? error.message : 'Something went wrong. Please try again.';
          patchMessage(replyId, (m) => ({ ...m, status: 'error', error: message }));
          if (error instanceof ApiError && error.code === 'unreachable') setConnection('offline');
          if (error instanceof ApiError && error.code === 'not_configured') setConnection('unconfigured');
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        streamingRef.current = false;
        setIsStreaming(false);
        setActivity(null);
      }
    },
    [patchMessage],
  );

  const send = useCallback(
    (raw: string) => {
      const text = raw.trim();
      if (!text || streamingRef.current) return;
      const now = Date.now();
      const question: ChatMessage = {
        id: uid(),
        role: 'user',
        content: text,
        createdAt: now,
        status: 'done',
        places: [],
        unresolved: [],
        error: null,
      };
      const reply: ChatMessage = {
        id: uid(),
        role: 'assistant',
        content: '',
        createdAt: now,
        status: 'streaming',
        places: [],
        unresolved: [],
        error: null,
      };
      const history = buildHistory(messagesRef.current);
      setMessages((prev) => [...prev, question, reply]);
      setActiveMessageId(reply.id);
      setSelectedPlaceId(null);
      setHoveredPlaceId(null);
      void runReply(reply.id, text, history);
    },
    [runReply],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const lastReply = useMemo(() => [...messages].reverse().find((m) => m.role === 'assistant') ?? null, [messages]);
  const canRetry = !isStreaming && !!lastReply && (lastReply.status === 'error' || lastReply.status === 'stopped');

  const retry = useCallback(() => {
    if (streamingRef.current) return;
    const all = messagesRef.current;
    const replyIndex = all.map((m) => m.role).lastIndexOf('assistant');
    const reply = all[replyIndex];
    if (!reply || (reply.status !== 'error' && reply.status !== 'stopped')) return;
    const questionIndex = all.slice(0, replyIndex).map((m) => m.role).lastIndexOf('user');
    const question = all[questionIndex];
    if (!question) return;
    const remaining = all.filter((_, i) => i !== replyIndex && i !== questionIndex);
    messagesRef.current = remaining;
    setMessages(remaining);
    send(question.content);
  }, [send]);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setMessages([]);
    setActiveMessageId(null);
    setSelectedPlaceId(null);
    setHoveredPlaceId(null);
    setActivity(null);
  }, []);

  // ---------------------------------------------------------------- places & map

  const activeMessage = useMemo(
    () => messages.find((m) => m.id === activeMessageId) ?? null,
    [messages, activeMessageId],
  );
  const places = useMemo(() => activeMessage?.places ?? [], [activeMessage]);
  const selectedPlace = useMemo(
    () => places.find((p) => p.id === selectedPlaceId) ?? null,
    [places, selectedPlaceId],
  );

  const focusMessage = useCallback((messageId: string) => {
    setActiveMessageId(messageId);
    setSelectedPlaceId(null);
  }, []);

  const selectPlace = useCallback((placeId: string | null) => {
    if (placeId) {
      const messageId = placeId.slice(0, placeId.lastIndexOf(':'));
      setActiveMessageId((current) => (current === messageId ? current : messageId));
    }
    setSelectedPlaceId(placeId);
  }, []);

  const hoverPlace = useCallback((placeId: string | null) => setHoveredPlaceId(placeId), []);

  const setMapView = useCallback((view: MapView) => {
    setMapViewState(view);
    writeJSON('view', view);
  }, []);

  const setUseMapArea = useCallback((value: boolean) => setUseMapAreaState(value), []);

  const registerMapDriver = useCallback((driver: MapDriver | null) => {
    driverRef.current = driver;
  }, []);

  const map = useMemo<MapApi>(
    () => ({
      zoomIn: () => driverRef.current?.zoomIn(),
      zoomOut: () => driverRef.current?.zoomOut(),
      fitToPlaces: () => driverRef.current?.fitToPlaces(),
      flyTo: (lat, lon, zoom) => driverRef.current?.flyTo(lat, lon, zoom),
      locate: () => {
        const current = contextRef.current.geo;
        if (current.status === 'on' && current.location) {
          driverRef.current?.flyTo(current.location.lat, current.location.lon, 15);
          return;
        }
        flyToFixRef.current = true;
        enableLocation();
      },
    }),
    [enableLocation],
  );

  const recheckConnection = useCallback(() => {
    setConnection('checking');
    void checkHealth();
  }, [checkHealth]);

  const value = useMemo<MapChatController>(
    () => ({
      messages,
      isStreaming,
      activity,
      send,
      stop,
      retry,
      canRetry,
      reset,
      activeMessageId,
      activeMessage,
      places,
      selectedPlace,
      selectedPlaceId,
      hoveredPlaceId,
      focusMessage,
      selectPlace,
      hoverPlace,
      mapView,
      setMapView,
      useMapArea,
      setUseMapArea,
      geo,
      enableLocation,
      disableLocation,
      connection,
      health,
      recheckConnection,
      map,
      registerMapDriver,
    }),
    [
      messages, isStreaming, activity, send, stop, retry, canRetry, reset, activeMessageId,
      activeMessage, places, selectedPlace, selectedPlaceId, hoveredPlaceId, focusMessage,
      selectPlace, hoverPlace, mapView, setMapView, useMapArea, setUseMapArea, geo,
      enableLocation, disableLocation, connection, health, recheckConnection, map, registerMapDriver,
    ],
  );

  return <MapChatContext.Provider value={value}>{children}</MapChatContext.Provider>;
}

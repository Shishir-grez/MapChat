/**
 * Sandbox: an intentionally plain developer view (dev builds only) that exercises every
 * controller feature. Also a compact reference for building a new variant.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import {
  MessageContent,
  PlaceMap,
  SUGGESTIONS,
  activityLabel,
  autosize,
  connectionHelp,
  connectionLabel,
  isSubmitKey,
  useMapChat,
  type PlaceRenderArgs,
} from '../../core';

export default function Sandbox() {
  const chat = useMapChat();
  const [draft, setDraft] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    autosize(inputRef.current);
  }, [draft]);
  useEffect(() => {
    // Block body: in current Chromium scrollIntoView() returns a Promise, which must not
    // be returned from an effect.
    endRef.current?.scrollIntoView({ block: 'end' });
  }, [chat.messages]);

  const submit = () => {
    if (!draft.trim() || chat.isStreaming) return;
    chat.send(draft);
    setDraft('');
  };

  const renderPlace = useCallback(
    ({ label, place, pending }: PlaceRenderArgs) =>
      place ? (
        <button
          type="button"
          className="mx-0.5 inline-flex items-baseline gap-1 rounded border border-line px-1 text-left hover:bg-ink/5"
          onClick={() => chat.selectPlace(place.id)}
          onMouseEnter={() => chat.hoverPlace(place.id)}
          onMouseLeave={() => chat.hoverPlace(null)}
        >
          <span className="font-mono text-xs">{place.index}</span>
          {label}
        </button>
      ) : (
        <span className={pending ? 'text-muted' : 'underline decoration-dotted'}>{label}</span>
      ),
    [chat],
  );

  const status = connectionLabel(chat.connection, chat.health);
  const help = connectionHelp(chat.connection);

  return (
    <div className="grid h-full grid-cols-1 grid-rows-[45%_1fr] md:grid-cols-[420px_1fr] md:grid-rows-1">
      <section className="order-2 flex min-h-0 flex-col border-r border-line md:order-1">
        <header className="flex items-center justify-between border-b border-line px-4 py-3 text-sm">
          <strong>MapChat · sandbox</strong>
          <span className="flex items-center gap-3">
            {status ? <span className="text-muted">{status}</span> : null}
            <button type="button" onClick={chat.reset}>
              New
            </button>
          </span>
        </header>

        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4 text-sm" aria-live="polite">
          {help ? <p className="rounded border border-line p-3 text-muted">{help}</p> : null}
          {chat.messages.length === 0 ? (
            <ul className="space-y-2">
              {SUGGESTIONS.map((s) => (
                <li key={s.title}>
                  <button type="button" className="text-left underline" onClick={() => chat.send(s.prompt)}>
                    {s.title}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
          {chat.messages.map((m) =>
            m.role === 'user' ? (
              <p key={m.id} className="whitespace-pre-wrap font-medium">
                {m.content}
              </p>
            ) : (
              <article
                key={m.id}
                className={m.id === chat.activeMessageId ? 'border-l-2 border-ink pl-3' : 'pl-3'}
                onClick={() => m.places.length && chat.focusMessage(m.id)}
              >
                <MessageContent message={m} renderPlace={renderPlace} className="space-y-2 [&_h3]:font-semibold [&_ul]:list-disc [&_ul]:pl-5" />
                {m.status === 'error' ? <p className="text-red-700">{m.error}</p> : null}
                {m.status === 'stopped' ? <p className="text-muted">Stopped.</p> : null}
                {m.unresolved.length ? <p className="text-faint">Not on the map: {m.unresolved.join('; ')}</p> : null}
              </article>
            ),
          )}
          {chat.activity ? <p className="text-muted">{activityLabel(chat.activity)}…</p> : null}
          {chat.canRetry ? (
            <button type="button" className="underline" onClick={chat.retry}>
              Retry
            </button>
          ) : null}
          <div ref={endRef} />
        </div>

        <form
          className="flex items-end gap-2 border-t border-line p-3"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <textarea
            ref={inputRef}
            rows={1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (isSubmitKey(e)) {
                e.preventDefault();
                submit();
              }
            }}
            aria-label="Message"
            placeholder="Ask about a place…"
            className="min-h-[38px] flex-1 resize-none rounded border border-line p-2 text-sm"
          />
          {chat.isStreaming ? (
            <button type="button" onClick={chat.stop} className="rounded border border-line px-3 py-2 text-sm">
              Stop
            </button>
          ) : (
            <button type="submit" disabled={!draft.trim()} className="rounded bg-ink px-3 py-2 text-sm text-canvas disabled:opacity-40">
              Send
            </button>
          )}
        </form>
        <div className="flex flex-wrap gap-3 px-3 pb-3 text-xs text-muted">
          <label className="flex items-center gap-1">
            <input type="checkbox" checked={chat.useMapArea} onChange={(e) => chat.setUseMapArea(e.target.checked)} />
            Use map area
          </label>
          <label className="flex items-center gap-1">
            <input
              type="checkbox"
              checked={chat.geo.status === 'on' || chat.geo.status === 'requesting'}
              onChange={(e) => (e.target.checked ? chat.enableLocation() : chat.disableLocation())}
            />
            Share my location {chat.geo.error ? `(${chat.geo.error})` : ''}
          </label>
        </div>
      </section>

      <section className="relative order-1 md:order-2">
        <PlaceMap
          tiles="positron"
          renderPopup={(p) => (
            <div className="text-sm">
              <strong>{p.name}</strong>
              {p.address ? <div className="text-muted">{p.address}</div> : null}
              <a href={p.mapsUrl} target="_blank" rel="noreferrer" className="underline">
                Open in Google Maps
              </a>
            </div>
          )}
        />
        <div className="absolute right-3 top-3 z-[500] flex flex-col gap-1 text-sm">
          <button type="button" className="rounded border border-line bg-white px-2" onClick={chat.map.zoomIn} aria-label="Zoom in">+</button>
          <button type="button" className="rounded border border-line bg-white px-2" onClick={chat.map.zoomOut} aria-label="Zoom out">−</button>
          <button type="button" className="rounded border border-line bg-white px-2" onClick={chat.map.fitToPlaces}>Fit</button>
          <button type="button" className="rounded border border-line bg-white px-2" onClick={chat.map.locate}>Me</button>
        </div>
      </section>
    </div>
  );
}

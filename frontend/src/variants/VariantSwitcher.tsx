import { useEffect, useRef, useState } from 'react';
import { VARIANTS, type VariantId } from './registry';

interface Props {
  current: VariantId;
  onChange: (id: VariantId) => void;
}

/** A small, neutral control on the right edge for comparing the UI variants. */
export function VariantSwitcher({ current, onChange }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false);
    const onClick = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener('keydown', onKey);
    window.addEventListener('mousedown', onClick);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('mousedown', onClick);
    };
  }, [open]);

  return (
    <div ref={rootRef} className="mc-switcher" data-open={open}>
      <button
        type="button"
        className="mc-switcher-tab"
        aria-expanded={open}
        aria-controls="mc-switcher-menu"
        onClick={() => setOpen((v) => !v)}
      >
        UI
      </button>
      {open ? (
        <div id="mc-switcher-menu" className="mc-switcher-menu" role="menu" aria-label="Interface variant">
          <p className="mc-switcher-heading">Interface</p>
          {VARIANTS.map((v, i) => (
            <button
              key={v.id}
              type="button"
              role="menuitemradio"
              aria-checked={v.id === current}
              className="mc-switcher-item"
              onClick={() => {
                onChange(v.id);
                setOpen(false);
              }}
            >
              <span className="mc-switcher-num">{String(i + 1).padStart(2, '0')}</span>
              <span>
                <span className="mc-switcher-name">{v.name}</span>
                <span className="mc-switcher-tagline">{v.tagline}</span>
              </span>
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

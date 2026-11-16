import { Component, lazy, Suspense, useEffect, useState, type ComponentType, type LazyExoticComponent, type ReactNode } from 'react';
import { MapChatProvider } from './core';
import { readJSON, removeKey, writeJSON } from './lib/storage';
import { DEFAULT_VARIANT, VARIANTS, findVariant, type VariantId } from './variants/registry';
import { VariantSwitcher } from './variants/VariantSwitcher';

const lazyVariants = new Map<VariantId, LazyExoticComponent<ComponentType>>();
function variantComponent(id: VariantId) {
  let component = lazyVariants.get(id);
  if (!component) {
    component = lazy(findVariant(id)!.load);
    lazyVariants.set(id, component);
  }
  return component;
}

function initialVariant(): VariantId {
  const fromUrl = new URLSearchParams(window.location.search).get('ui');
  return (
    findVariant(fromUrl)?.id ??
    findVariant(readJSON<string | null>('variant', null))?.id ??
    findVariant(import.meta.env.VITE_DEFAULT_VARIANT)?.id ??
    DEFAULT_VARIANT
  );
}

const showSwitcher =
  import.meta.env.VITE_VARIANT_SWITCHER !== 'off' &&
  new URLSearchParams(window.location.search).get('switcher') !== '0' &&
  VARIANTS.length > 1;

class CrashBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() {
    return { failed: true };
  }
  render() {
    if (!this.state.failed) return this.props.children;
    return (
      <div className="mc-crash" role="alert">
        <p>Something went wrong while drawing the page.</p>
        <button
          type="button"
          onClick={() => {
            removeKey('conversation');
            window.location.reload();
          }}
        >
          Clear the conversation and reload
        </button>
      </div>
    );
  }
}

export default function App() {
  const [variant, setVariant] = useState<VariantId>(initialVariant);
  const Variant = variantComponent(variant);

  useEffect(() => {
    writeJSON('variant', variant);
    document.documentElement.dataset.variant = variant;
    const url = new URL(window.location.href);
    if (url.searchParams.get('ui') !== variant) {
      url.searchParams.set('ui', variant);
      window.history.replaceState(null, '', url);
    }
  }, [variant]);

  return (
    <MapChatProvider>
      <CrashBoundary key={variant}>
        <Suspense fallback={<div className="h-full w-full bg-white" />}>
          <Variant />
        </Suspense>
      </CrashBoundary>
      {showSwitcher ? <VariantSwitcher current={variant} onChange={setVariant} /> : null}
    </MapChatProvider>
  );
}

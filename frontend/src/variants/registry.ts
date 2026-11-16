import type { ComponentType } from 'react';

export type VariantId = 'atlas' | 'meridian' | 'concierge' | 'sandbox';

export interface VariantInfo {
  id: VariantId;
  name: string;
  tagline: string;
  load: () => Promise<{ default: ComponentType }>;
}

export const VARIANTS: VariantInfo[] = [
  {
    id: 'atlas',
    name: 'Atlas',
    tagline: 'Editorial — a serif reading column beside a monochrome map',
    load: () => import('./atlas'),
  },
  {
    id: 'meridian',
    name: 'Meridian',
    tagline: 'Map-first — a precise floating panel over a full-bleed map',
    load: () => import('./meridian'),
  },
  {
    id: 'concierge',
    name: 'Concierge',
    tagline: 'Hospitality — warm paper tones and an itinerary beside a framed map',
    load: () => import('./concierge'),
  },
  ...(import.meta.env.DEV
    ? [
        {
          id: 'sandbox' as const,
          name: 'Sandbox',
          tagline: 'Unstyled developer view of every feature',
          load: () => import('./sandbox'),
        },
      ]
    : []),
];

export const DEFAULT_VARIANT: VariantId = 'atlas';

export function findVariant(id: string | null | undefined): VariantInfo | undefined {
  return VARIANTS.find((v) => v.id === id);
}

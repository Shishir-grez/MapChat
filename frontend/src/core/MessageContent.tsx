/**
 * Safe markdown rendering of an assistant reply, with [LOCATION] tags turned into
 * whatever the variant renders for a place (usually a numbered, clickable chip).
 *
 * Raw HTML in replies is never rendered (react-markdown escapes it; skipHtml drops it).
 */
import { memo, useMemo, type ReactNode } from 'react';
import Markdown, { type Components } from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { PLACE_HREF_PREFIX, labelKey, toRenderableMarkdown } from '../lib/locations';
import type { ChatMessage, Place } from '../lib/types';

export interface PlaceRenderArgs {
  /** The tag text, e.g. "Shaniwar Wada, Shivajinagar, Pune". */
  label: string;
  /** The geocoded place, or null if not found (yet). */
  place: Place | null;
  /** True while the reply streams and this place has not been located yet. */
  pending: boolean;
  /** Occurrence number of this tag within the message (unique per message). */
  occurrence: number;
}

export interface MessageContentProps {
  message: ChatMessage;
  renderPlace: (args: PlaceRenderArgs) => ReactNode;
  className?: string;
  /** Override renderers for other markdown elements (headings, lists, tables…). */
  components?: Components;
}

const remarkPlugins = [remarkGfm];

function MessageContentImpl({ message, renderPlace, className = 'mc-prose', components }: MessageContentProps) {
  const streaming = message.status === 'streaming';
  const { markdown, labels } = useMemo(
    () => toRenderableMarkdown(message.content, streaming),
    [message.content, streaming],
  );
  const placesByKey = useMemo(
    () => new Map(message.places.map((p) => [labelKey(p.label), p] as const)),
    [message.places],
  );

  const merged = useMemo<Components>(
    () => ({
      ...components,
      a: ({ href, children, node: _node, ...rest }) => {
        if (href?.startsWith(PLACE_HREF_PREFIX)) {
          const occurrence = Number(href.slice(PLACE_HREF_PREFIX.length));
          const label = labels[occurrence];
          if (label !== undefined) {
            const place = placesByKey.get(labelKey(label)) ?? null;
            return <>{renderPlace({ label, place, pending: streaming && !place, occurrence })}</>;
          }
        }
        return (
          <a href={href} target="_blank" rel="noopener noreferrer nofollow" {...rest}>
            {children}
          </a>
        );
      },
    }),
    [components, labels, placesByKey, renderPlace, streaming],
  );

  return (
    <div className={className}>
      <Markdown remarkPlugins={remarkPlugins} components={merged} skipHtml>
        {markdown}
      </Markdown>
    </div>
  );
}

export const MessageContent = memo(MessageContentImpl);

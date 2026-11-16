/**
 * Parsing of the [LOCATION]…[/LOCATION] markup in assistant replies.
 * Mirrors backend/locations.py — keep the normalization rules in sync.
 */

// A complete tag whose content contains no other opening/closing tag.
const TAG_RE = /\[LOCATION\]((?:(?!\[\/?LOCATION\])[\s\S])*?)\[\/LOCATION\]/gi;
const ORPHAN_RE = /\[\/?LOCATION\]/gi;
const OPEN_TAG_RE = /\[LOCATION\]/gi;
// The tail of a stream that may be the beginning of a tag: "[", "[LOC", "[/LOCATIO", …
const PARTIAL_TAG_RE = /\[\/?(?:L(?:O(?:C(?:A(?:T(?:I(?:O(?:N)?)?)?)?)?)?)?)?$/i;
const MAX_LABEL_CHARS = 200;

/** Link target used to smuggle a tag through markdown: `#mc-place-<n>`. */
export const PLACE_HREF_PREFIX = '#mc-place-';

export function normalizeLabel(raw: string): string {
  let label = raw.split('**').join('').split('__').join('').split('`').join('');
  label = label.replace(/\s+/g, ' ').trim();
  label = label.replace(/^[*_ ]+|[*_ ]+$/g, '');
  label = label.replace(/[,;: ]+$/, '');
  return label;
}

/** Case-insensitive identity used to match chips to places. */
export function labelKey(label: string): string {
  return normalizeLabel(label).toLowerCase();
}

/** "Shaniwar Wada, Shivajinagar, Pune" → "Shaniwar Wada" */
export function displayName(label: string): string {
  const head = label.split(',', 1)[0]?.trim();
  return head || label;
}

/** Unique, normalized labels in order of first appearance. */
export function extractLocations(text: string): string[] {
  const seen = new Set<string>();
  const labels: string[] = [];
  for (const match of text.matchAll(TAG_RE)) {
    const label = normalizeLabel(match[1] ?? '');
    if (!label || label.length > MAX_LABEL_CHARS) continue;
    const key = label.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    labels.push(label);
  }
  return labels;
}

/** Plain text for copying: tags replaced by their content. */
export function stripLocationTags(text: string): string {
  return text.replace(TAG_RE, (_m, inner: string) => inner.trim()).replace(ORPHAN_RE, '');
}

/**
 * While a reply is streaming, hide a tag that has started but not finished:
 * the label text shows as plain text until the closing tag arrives.
 */
export function settleStreamingTail(text: string): string {
  let out = text.replace(PARTIAL_TAG_RE, '');
  let lastOpen = -1;
  for (const m of out.matchAll(OPEN_TAG_RE)) lastOpen = m.index ?? -1;
  if (lastOpen >= 0 && !/\[\/LOCATION\]/i.test(out.slice(lastOpen))) {
    out = out.slice(0, lastOpen) + out.slice(lastOpen + '[LOCATION]'.length);
  }
  return out;
}

export interface RenderableMarkdown {
  /** Markdown in which every tag became a link `[…](#mc-place-<n>)`. */
  markdown: string;
  /** labels[n] is the label of the n-th tag occurrence. */
  labels: string[];
}

function escapeLinkText(text: string): string {
  return text.replace(/([\\[\]*_`<>])/g, '\\$1');
}

export function toRenderableMarkdown(content: string, streaming = false): RenderableMarkdown {
  const source = streaming ? settleStreamingTail(content) : content;
  const labels: string[] = [];
  const replaced = source.replace(TAG_RE, (_whole, inner: string) => {
    const label = normalizeLabel(inner);
    if (!label || label.length > MAX_LABEL_CHARS) return inner;
    labels.push(label);
    return `[${escapeLinkText(label)}](${PLACE_HREF_PREFIX}${labels.length - 1})`;
  });
  return { markdown: replaced.replace(ORPHAN_RE, ''), labels };
}

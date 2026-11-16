"""Parsing of the [LOCATION]...[/LOCATION] markup the model uses to mark mappable places.

The frontend (src/lib/locations.ts) mirrors these rules so that labels produced here
match the location chips it renders. Keep the two in sync.
"""

from __future__ import annotations

import re
from urllib.parse import quote

# A complete tag whose content does not itself contain an opening/closing tag. The
# tempered token means "[LOCATION]a [LOCATION]b[/LOCATION]" yields "b", not "a [LOCATION]b".
TAG_RE = re.compile(
    r"\[LOCATION\]((?:(?!\[/?LOCATION\]).)*?)\[/LOCATION\]",
    re.IGNORECASE | re.DOTALL,
)
ORPHAN_TAG_RE = re.compile(r"\[/?LOCATION\]", re.IGNORECASE)
MAX_LABEL_CHARS = 200

_WS_RE = re.compile(r"\s+")


def normalize_label(raw: str) -> str:
    """Clean a tag's inner text: drop markdown emphasis, collapse whitespace, trim."""
    label = raw.replace("**", "").replace("__", "").replace("`", "")
    label = _WS_RE.sub(" ", label).strip()
    label = label.strip("*_ ")
    label = label.rstrip(",;: ")
    return label


def label_key(label: str) -> str:
    """Case-insensitive identity used to de-duplicate places."""
    return normalize_label(label).casefold()


def display_name(label: str) -> str:
    """'Shaniwar Wada, Shivajinagar, Pune' -> 'Shaniwar Wada'."""
    head = label.split(",", 1)[0].strip()
    return head or label


def _valid(label: str) -> bool:
    return 0 < len(label) <= MAX_LABEL_CHARS


def extract_locations(text: str, limit: int | None = None) -> list[str]:
    """Return unique, normalized labels in order of first appearance."""
    tracker = LocationTracker(limit=limit)
    return tracker.feed(text)


def strip_location_tags(text: str) -> str:
    """Replace tags with their inner text and drop any unmatched tag markers."""
    stripped = TAG_RE.sub(lambda m: m.group(1).strip(), text)
    return ORPHAN_TAG_RE.sub("", stripped)


class LocationTracker:
    """Incrementally finds newly completed tags in a growing (streamed) buffer."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.labels: list[str] = []
        self._seen: set[str] = set()
        self._pos = 0

    def feed(self, text: str) -> list[str]:
        """Scan `text` (the full buffer so far) and return labels not seen before."""
        new: list[str] = []
        for match in TAG_RE.finditer(text, self._pos):
            self._pos = match.end()
            label = normalize_label(match.group(1))
            if not _valid(label):
                continue
            key = label.casefold()
            if key in self._seen:
                continue
            if self.limit is not None and len(self.labels) >= self.limit:
                continue
            self._seen.add(key)
            self.labels.append(label)
            new.append(label)
        return new


def maps_url(label: str, lat: float, lon: float, place_id: str | None = None) -> str:
    """A Google Maps link that works without an API key."""
    if place_id:
        return (
            "https://www.google.com/maps/search/?api=1"
            f"&query={quote(label)}&query_place_id={quote(place_id)}"
        )
    return f"https://www.google.com/maps/search/?api=1&query={lat:.6f},{lon:.6f}"

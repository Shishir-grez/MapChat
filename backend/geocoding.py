"""Geocoding back-ends (Google Maps, OpenStreetMap Nominatim, demo) behind one interface."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Hashable, Protocol

from locations import label_key

if TYPE_CHECKING:
    from config import Settings

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeoResult:
    lat: float
    lon: float
    address: str | None = None
    place_id: str | None = None
    approximate: bool = False


class GeocodingError(Exception):
    """A transport or provider failure. Unlike 'not found', it is never cached."""


class Geocoder(Protocol):
    name: str

    def geocode(self, query: str) -> GeoResult | None: ...

    def reverse(self, lat: float, lon: float) -> str | None: ...


class GoogleGeocoder:
    name = "google"

    def __init__(self, api_key: str, client: Any | None = None) -> None:
        if client is None:
            import googlemaps

            client = googlemaps.Client(key=api_key, timeout=10, retry_timeout=10)
        self._client = client

    def geocode(self, query: str) -> GeoResult | None:
        try:
            results = self._client.geocode(query)
        except Exception as exc:  # googlemaps raises several unrelated exception types
            raise GeocodingError(f"Google geocoding failed: {exc}") from exc
        if not results:
            return None
        top = results[0]
        loc = top["geometry"]["location"]
        return GeoResult(
            lat=float(loc["lat"]),
            lon=float(loc["lng"]),
            address=top.get("formatted_address"),
            place_id=top.get("place_id"),
            approximate=bool(top.get("partial_match")),
        )

    def reverse(self, lat: float, lon: float) -> str | None:
        try:
            results = self._client.reverse_geocode(
                (lat, lon), result_type=["neighborhood", "sublocality", "locality"]
            )
        except Exception as exc:
            raise GeocodingError(f"Google reverse geocoding failed: {exc}") from exc
        if not results:
            return None
        return results[0].get("formatted_address")


class NominatimGeocoder:
    """OpenStreetMap's free geocoder. Its usage policy allows at most 1 request/second."""

    name = "nominatim"
    min_interval = 1.0

    def __init__(self, user_agent: str, client: Any | None = None) -> None:
        if client is None:
            from geopy.geocoders import Nominatim

            client = Nominatim(user_agent=user_agent, timeout=10)
        self._client = client
        self._lock = threading.Lock()
        self._last_call = 0.0

    def _call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                raise GeocodingError(f"Nominatim request failed: {exc}") from exc
            finally:
                self._last_call = time.monotonic()

    def geocode(self, query: str) -> GeoResult | None:
        location = self._call(self._client.geocode, query, exactly_one=True)
        approximate = False
        parts = [p.strip() for p in query.split(",") if p.strip()]
        if location is None and len(parts) >= 3:
            # Nominatim is weak on business names; fall back to the street/area.
            location = self._call(self._client.geocode, ", ".join(parts[1:]), exactly_one=True)
            approximate = location is not None
        if location is None:
            return None
        return GeoResult(
            lat=float(location.latitude),
            lon=float(location.longitude),
            address=getattr(location, "address", None),
            approximate=approximate,
        )

    def reverse(self, lat: float, lon: float) -> str | None:
        location = self._call(self._client.reverse, (lat, lon), zoom=14, exactly_one=True)
        if location is None:
            return None
        address = (getattr(location, "raw", None) or {}).get("address") or {}
        parts = [
            address.get("suburb") or address.get("neighbourhood") or address.get("quarter"),
            address.get("city") or address.get("town") or address.get("village"),
            address.get("state"),
            address.get("country"),
        ]
        short = ", ".join(dict.fromkeys(p for p in parts if p))
        return short or getattr(location, "address", None)


class _LRU:
    """A small thread-safe LRU cache that can store None (a cached 'not found')."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[Hashable, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> tuple[bool, Any]:
        with self._lock:
            if key not in self._data:
                return False, None
            self._data.move_to_end(key)
            return True, self._data[key]

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


class CachingGeocoder:
    """Caches results per normalized query; provider failures are logged, not cached."""

    def __init__(self, inner: Geocoder, maxsize: int = 2048) -> None:
        self.inner = inner
        self.name = inner.name
        self._forward = _LRU(maxsize)
        self._reverse = _LRU(512)

    def geocode(self, query: str) -> GeoResult | None:
        key = label_key(query)
        hit, value = self._forward.get(key)
        if hit:
            return value
        try:
            value = self.inner.geocode(query)
        except GeocodingError as exc:
            log.warning("%s", exc)
            return None
        self._forward.set(key, value)
        return value

    def reverse(self, lat: float, lon: float) -> str | None:
        # ~1 km buckets: plenty for describing "the area the user is looking at".
        key = (round(lat, 2), round(lon, 2))
        hit, value = self._reverse.get(key)
        if hit:
            return value
        try:
            value = self.inner.reverse(lat, lon)
        except GeocodingError as exc:
            log.warning("%s", exc)
            return None
        self._reverse.set(key, value)
        return value


def make_geocoder(settings: "Settings") -> CachingGeocoder:
    inner: Geocoder
    if settings.demo:
        from demo import DemoGeocoder

        inner = DemoGeocoder()
    elif settings.google_maps_api_key:
        inner = GoogleGeocoder(settings.google_maps_api_key)
    else:
        log.info("GOOGLE_MAPS_API_KEY not set; using OpenStreetMap Nominatim (1 request/second).")
        inner = NominatimGeocoder(settings.nominatim_user_agent)
    return CachingGeocoder(inner)

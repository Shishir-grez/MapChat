"""MapChat HTTP API.

Endpoints
  GET  /api/health        configuration summary
  POST /api/chat          JSON reply: {content, places, unresolved}
  POST /api/chat/stream   the same reply as Server-Sent Events (status/delta/place/done/error)

Run with `python server.py` (see README.md for configuration).
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.security import safe_join

from agent import ChatBackend, HistoryMessage, describe_error, make_chat_backend
from config import BACKEND_DIR, Settings
from geocoding import Geocoder, make_geocoder
from locations import LocationTracker, display_name, maps_url
from ratelimit import SlidingWindowLimiter

log = logging.getLogger("mapchat")

_GEOCODE_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="geocode")
MIN_CONTEXT_ZOOM = 10  # below this the map shows a region, not a neighbourhood


# --------------------------------------------------------------------------- requests


class RequestError(Exception):
    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        field: str | None = None,
        status: int = 400,
        retry_after: int | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.field = field
        self.status = status
        self.retry_after = retry_after


@dataclass(frozen=True)
class MapContext:
    lat: float
    lon: float
    zoom: float | None = None


@dataclass(frozen=True)
class UserLocation:
    lat: float
    lon: float
    accuracy: float | None = None


@dataclass(frozen=True)
class ChatRequest:
    message: str
    history: tuple[HistoryMessage, ...] = ()
    map: MapContext | None = None
    user_location: UserLocation | None = None


def _number(value: Any, field: str, lo: float | None = None, hi: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RequestError(f"'{field}' must be a number.", field=field)
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise RequestError(f"'{field}' must be between {lo} and {hi}.", field=field)
    return float(value)


def _wrap_lon(lon: float) -> float:
    return ((lon + 180.0) % 360.0) - 180.0


def _point(raw: Any, field: str) -> tuple[float, float]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RequestError(f"'{field}' must be [lat, lon].", field=field)
    lat = _number(raw[0], f"{field}[0]", -90, 90)
    lon = _wrap_lon(_number(raw[1], f"{field}[1]"))
    return lat, lon


def parse_chat_request(payload: Any, settings: Settings) -> ChatRequest:
    if not isinstance(payload, dict):
        raise RequestError("Request body must be a JSON object.", code="invalid_json")

    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        raise RequestError("'message' must be a non-empty string.", field="message")
    message = message.strip()
    if len(message) > settings.max_message_chars:
        raise RequestError(
            f"'message' is too long (max {settings.max_message_chars} characters).", field="message"
        )

    raw_history = payload.get("history") or []
    if not isinstance(raw_history, list) or len(raw_history) > 200:
        raise RequestError("'history' must be a list of messages.", field="history")
    history: list[HistoryMessage] = []
    for i, item in enumerate(raw_history):
        role = item.get("role") if isinstance(item, dict) else None
        content = item.get("content") if isinstance(item, dict) else None
        if role not in ("user", "assistant") or not isinstance(content, str):
            raise RequestError(
                f"'history[{i}]' needs a role of 'user' or 'assistant' and string content.",
                field="history",
            )
        content = content.strip()
        if content:
            history.append(HistoryMessage(role=role, content=content[: settings.max_history_chars]))
    keep = settings.max_history_messages
    history = history[-keep:] if keep > 0 else []

    context = payload.get("context") or {}
    if not isinstance(context, dict):
        raise RequestError("'context' must be an object.", field="context")

    map_ctx: MapContext | None = None
    raw_map = context.get("map")
    if raw_map is not None:
        if not isinstance(raw_map, dict):
            raise RequestError("'context.map' must be an object.", field="context.map")
        lat, lon = _point(raw_map.get("center"), "context.map.center")
        zoom = raw_map.get("zoom")
        map_ctx = MapContext(lat, lon, None if zoom is None else _number(zoom, "context.map.zoom", 0, 24))

    user_location: UserLocation | None = None
    raw_loc = context.get("user_location")
    if raw_loc is not None:
        if not isinstance(raw_loc, dict):
            raise RequestError("'context.user_location' must be an object.", field="context.user_location")
        lat = _number(raw_loc.get("lat"), "context.user_location.lat", -90, 90)
        lon = _wrap_lon(_number(raw_loc.get("lon"), "context.user_location.lon"))
        accuracy = raw_loc.get("accuracy")
        user_location = UserLocation(
            lat, lon, None if accuracy is None else _number(accuracy, "context.user_location.accuracy", 0)
        )

    return ChatRequest(message, tuple(history), map_ctx, user_location)


# --------------------------------------------------------------------------- chat pipeline


def build_context(req: ChatRequest, geocoder: Geocoder) -> list[str]:
    """Describe where the user is / is looking, in words the model can use."""
    lines: list[str] = []
    if req.user_location:
        loc = req.user_location
        area = geocoder.reverse(loc.lat, loc.lon)
        accuracy = f", ±{loc.accuracy:.0f} m" if loc.accuracy else ""
        where = f"{area} " if area else ""
        lines.append(
            f"The user shared their current location: {where}({loc.lat:.5f}, {loc.lon:.5f}{accuracy}). "
            "Interpret 'near me', 'nearby' and 'around here' relative to this point."
        )
    if req.map and (req.map.zoom is None or req.map.zoom >= MIN_CONTEXT_ZOOM):
        area = geocoder.reverse(req.map.lat, req.map.lon)
        zoom = f", zoom {req.map.zoom:.0f}" if req.map.zoom is not None else ""
        lines.append(
            f"The user's map is centred on {area or 'an unnamed area'} "
            f"({req.map.lat:.4f}, {req.map.lon:.4f}{zoom}). "
            "Treat it as the default area when the request doesn't name one."
        )
    return lines


class PlaceResolver:
    """Geocodes tags as soon as they complete and releases results in reading order."""

    def __init__(self, geocoder: Geocoder, limit: int) -> None:
        self.geocoder = geocoder
        self.tracker = LocationTracker(limit=limit)
        self.pending: list[tuple[str, Future]] = []
        self.places: list[dict[str, Any]] = []
        self.unresolved: list[str] = []

    def feed(self, text: str) -> None:
        for label in self.tracker.feed(text):
            self.pending.append((label, _GEOCODE_POOL.submit(self.geocoder.geocode, label)))

    def drain(self, *, block: bool) -> Iterator[dict[str, Any]]:
        while self.pending:
            label, future = self.pending[0]
            if not block and not future.done():
                return
            self.pending.pop(0)
            try:
                geo = future.result()
            except Exception:  # a geocoder bug must not kill the reply
                log.exception("Geocoding %r failed", label)
                geo = None
            if geo is None:
                self.unresolved.append(label)
                continue
            place = {
                "index": len(self.places) + 1,
                "label": label,
                "name": display_name(label),
                "address": geo.address,
                "lat": round(geo.lat, 6),
                "lon": round(geo.lon, 6),
                "approximate": geo.approximate,
                "maps_url": maps_url(label, geo.lat, geo.lon, geo.place_id),
            }
            self.places.append(place)
            yield place

    def cancel(self) -> None:
        for _, future in self.pending:
            future.cancel()


def chat_events(
    req: ChatRequest, backend: ChatBackend, geocoder: Geocoder, settings: Settings
) -> Iterator[tuple[str, dict[str, Any]]]:
    resolver = PlaceResolver(geocoder, settings.max_places)
    text = ""
    try:
        yield "status", {"stage": "thinking", "detail": None}
        context = build_context(req, geocoder)
        for event in backend.stream(req.message, req.history, context):
            if event.kind == "delta":
                text += event.text
                yield "delta", {"text": event.text}
                resolver.feed(text)
                for place in resolver.drain(block=False):
                    yield "place", place
            else:
                yield "status", {"stage": event.stage, "detail": event.detail}
        if resolver.pending:
            count = len(resolver.pending)
            yield "status", {
                "stage": "locating",
                "detail": f"Pinning {count} place{'s' if count != 1 else ''} on the map",
            }
        for place in resolver.drain(block=True):
            yield "place", place
        yield "done", {"content": text, "places": resolver.places, "unresolved": resolver.unresolved}
    except Exception as exc:
        code, message = describe_error(exc)
        log.exception("Chat request failed (%s)", code)
        yield "error", {"code": code, "message": message}
    finally:
        resolver.cancel()


def sse(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event}\ndata: {payload}\n\n"


# --------------------------------------------------------------------------- app


def create_app(
    settings: Settings | None = None,
    *,
    chat_backend: ChatBackend | None = None,
    geocoder: Geocoder | None = None,
) -> Flask:
    if settings is None:
        from dotenv import load_dotenv

        load_dotenv(BACKEND_DIR / ".env")
        settings = Settings.from_env()

    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = 512 * 1024
    app.json.ensure_ascii = False  # type: ignore[attr-defined]
    if settings.trust_proxy:
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)  # type: ignore[method-assign]

    origins: str | list[str] = "*" if "*" in settings.cors_origins else list(settings.cors_origins)
    CORS(app, resources={r"/api/*": {"origins": origins}})

    backend = chat_backend if chat_backend is not None else make_chat_backend(settings)
    geo = geocoder if geocoder is not None else make_geocoder(settings)
    limiter = SlidingWindowLimiter(settings.rate_limit_per_minute)

    if backend is None:
        log.warning(
            "OPENAI_API_KEY is not set: chat is disabled. Add it to backend/.env, "
            "or set MAPCHAT_DEMO=1 to try the app with canned answers."
        )

    def error_response(code: str, message: str, status: int, **extra: Any) -> Response:
        body: dict[str, Any] = {"code": code, "message": message}
        body.update({k: v for k, v in extra.items() if v is not None})
        response = jsonify({"error": body})
        response.status_code = status
        return response

    @app.errorhandler(RequestError)
    def _request_error(exc: RequestError) -> Response:
        response = error_response(exc.code, exc.message, exc.status, field=exc.field)
        if exc.retry_after is not None:
            response.headers["Retry-After"] = str(exc.retry_after)
        return response

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException) -> Any:
        if not request.path.startswith("/api/"):
            return exc
        code = (exc.name or "error").lower().replace(" ", "_")
        return error_response(code, exc.description or exc.name, exc.code or 500)

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception) -> Response:
        log.exception("Unhandled error on %s", request.path)
        return error_response("internal", "Internal server error.", 500)

    def prepare() -> ChatRequest:
        payload = request.get_json(silent=True)
        if payload is None:
            raise RequestError(
                "Request body must be JSON (Content-Type: application/json).", code="invalid_json"
            )
        chat_request = parse_chat_request(payload, settings)
        if backend is None:
            raise RequestError(
                "The chat model isn't configured. Set OPENAI_API_KEY in backend/.env "
                "or run the server with MAPCHAT_DEMO=1.",
                code="not_configured",
                status=503,
            )
        retry_after = limiter.hit(request.remote_addr or "unknown")
        if retry_after is not None:
            raise RequestError(
                "Too many requests. Please wait a moment and try again.",
                code="rate_limited",
                status=429,
                retry_after=max(1, math.ceil(retry_after)),
            )
        return chat_request

    def log_outcome(rid: str, req: ChatRequest, started: float, outcome: str, places: int) -> None:
        log.info(
            "chat %s outcome=%s chars=%d history=%d places=%d map=%s location=%s %.2fs",
            rid, outcome, len(req.message), len(req.history), places,
            req.map is not None, req.user_location is not None, time.monotonic() - started,
        )

    @app.get("/api/health")
    def health() -> Response:
        return jsonify(
            {
                "status": "ok",
                "demo": settings.demo,
                "llm": backend is not None,
                "model": settings.openai_model if backend is not None and not settings.demo else None,
                "search": settings.search_enabled,
                "geocoder": geo.name,
            }
        )

    @app.post("/api/chat")
    def chat() -> Any:
        req = prepare()
        rid, started = uuid.uuid4().hex[:8], time.monotonic()
        result: dict[str, Any] | None = None
        for event, data in chat_events(req, backend, geo, settings):  # type: ignore[arg-type]
            if event == "error":
                log_outcome(rid, req, started, data["code"], 0)
                return error_response(data["code"], data["message"], 502)
            if event == "done":
                result = data
        assert result is not None
        log_outcome(rid, req, started, "ok", len(result["places"]))
        return jsonify(result)

    @app.post("/api/chat/stream")
    def chat_stream() -> Response:
        req = prepare()
        rid, started = uuid.uuid4().hex[:8], time.monotonic()

        def generate() -> Iterator[str]:
            outcome, places = "disconnected", 0
            try:
                yield ": stream open\n\n"
                for event, data in chat_events(req, backend, geo, settings):  # type: ignore[arg-type]
                    if event == "error":
                        outcome = data["code"]
                    elif event == "done":
                        outcome, places = "ok", len(data["places"])
                    yield sse(event, data)
            finally:
                log_outcome(rid, req, started, outcome, places)

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
        )

    # Serve the built frontend (frontend/dist) when it exists: one process in production.
    static_dir = Path(settings.static_dir) if settings.static_dir else None
    has_frontend = bool(static_dir and (static_dir / "index.html").is_file())

    @app.get("/")
    @app.get("/<path:path>")
    def frontend(path: str = "") -> Any:
        if path.startswith("api/") or path == "api":
            abort(404)
        if not has_frontend:
            if path:
                abort(404)
            return jsonify(
                {
                    "name": "MapChat API",
                    "health": "/api/health",
                    "hint": "Run `npm run dev` in frontend/, or `npm run build` to serve the UI from here.",
                }
            )
        assert static_dir is not None
        if path:
            target = safe_join(str(static_dir), path)
            if target is not None and Path(target).is_file():
                max_age = 31_536_000 if path.startswith("assets/") else 0
                return send_from_directory(static_dir, path, max_age=max_age)
            if "." in Path(path).name:
                abort(404)  # a missing asset, not a client-side route
        response = send_from_directory(static_dir, "index.html", max_age=0)
        response.headers["Cache-Control"] = "no-cache"
        return response

    return app


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv(BACKEND_DIR / ".env")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings.from_env()
    app = create_app(settings)
    mode = "demo" if settings.demo else ("ready" if settings.openai_api_key else "chat disabled")
    log.info(
        "MapChat API on http://%s:%d (%s; geocoder=%s; web search=%s)",
        settings.host, settings.port, mode, settings.geocoder_name,
        "on" if settings.search_enabled else "off",
    )
    app.run(host=settings.host, port=settings.port, debug=settings.debug, threaded=True)


if __name__ == "__main__":
    main()

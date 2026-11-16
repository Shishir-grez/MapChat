"""Chat back-ends. The real one drives an agno Agent (OpenAI + optional SerpAPI search).

A fresh Agent is built for every request and the conversation history is supplied by the
client, so no state is shared between users or requests.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal, Protocol, Sequence

if TYPE_CHECKING:
    from config import Settings

log = logging.getLogger(__name__)

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class HistoryMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class StreamEvent:
    """Either a chunk of reply text ("delta") or a progress update ("status")."""

    kind: Literal["delta", "status"]
    text: str = ""
    stage: str | None = None
    detail: str | None = None


class ChatBackend(Protocol):
    def stream(
        self, message: str, history: Sequence[HistoryMessage], context: Sequence[str]
    ) -> Iterator[StreamEvent]: ...


SYSTEM_PROMPT = """\
You are MapChat, a location-intelligence assistant. You answer questions about places, \
neighbourhoods, routes and travel plans. Every reply is shown next to an interactive map.

LOCATION TAGS (highest priority)
- Wrap every specific, mappable place you mention in [LOCATION]...[/LOCATION] tags. Each tag \
becomes a numbered pin on the user's map and a clickable chip in your reply.
- Inside a tag, write the place name followed by enough context to find it unambiguously: \
street or neighbourhood, then city (and country when it is outside the user's region). \
Examples: [LOCATION]Shaniwar Wada, Shivajinagar, Pune[/LOCATION], \
[LOCATION]Café de Flore, Saint-Germain-des-Prés, Paris[/LOCATION].
- Use exactly the same text inside the tag every time you mention the same place.
- Tag cities, districts and natural features too when they are a subject of the answer: \
[LOCATION]Lonavala, Maharashtra[/LOCATION].
- Never put markdown, links or another tag inside a tag. Do not tag vague references \
("a café nearby", "the north") or non-physical things.

ANSWER STYLE
- Lead with the direct answer, then supporting detail. Keep replies scannable: short \
paragraphs, bullet lists, **bold** for key facts. Use ### headings only for multi-part answers \
such as itineraries ("### Day 1").
- For each recommendation add a one-line reason: what it is known for, best time to go, price \
level or practical tips such as travel time between stops.
- When the request is ambiguous about location, use the context below. If there is none, \
assume the most likely place, say so briefly, and answer.

FRESHNESS
{freshness}

CONTEXT
- Today is {today}.
{context}
"""

FRESHNESS_WITH_SEARCH = """\
- Use the web search tool only for time-sensitive facts: opening hours, new or closed venues, \
events, prices, ratings. Answer everything else from your own knowledge.
- Flag details that may have changed recently."""

FRESHNESS_WITHOUT_SEARCH = """\
- You have no live web access. Answer from your own knowledge and say when details such as \
opening hours or prices may be out of date."""


def build_system_prompt(
    context: Sequence[str], *, search_enabled: bool, today: dt.date | None = None
) -> str:
    today = today or dt.date.today()
    context_text = "\n".join(f"- {line}" for line in context) or "- No map or location context."
    return SYSTEM_PROMPT.format(
        freshness=FRESHNESS_WITH_SEARCH if search_enabled else FRESHNESS_WITHOUT_SEARCH,
        today=today.strftime("%A, %d %B %Y"),
        context=context_text,
    )


def _search_detail(chunk: Any) -> str:
    tools = getattr(chunk, "tools", None) or []
    last = tools[-1] if tools else {}
    args = last.get("tool_args") if isinstance(last, dict) else None
    query = args.get("query") if isinstance(args, dict) else None
    if isinstance(query, str) and query.strip():
        query = query.strip()
        if len(query) > 80:
            query = query[:79] + "…"
        return f"Searching the web for “{query}”"
    return "Searching the web"


class AgnoChatBackend:
    def __init__(self, settings: "Settings", base_url: str | None = None) -> None:
        self.settings = settings
        self.base_url = base_url  # lets tests point the OpenAI client at a local fake

    def _build_agent(self, history: Sequence[HistoryMessage], context: Sequence[str]) -> Any:
        from agno.agent import Agent
        from agno.models.message import Message
        from agno.models.openai import OpenAIChat

        tools: list[Any] = []
        if self.settings.search_enabled:
            from agno.tools.serpapi import SerpApiTools

            tools.append(SerpApiTools(api_key=self.settings.serpapi_key))

        model_kwargs: dict[str, Any] = {
            "id": self.settings.openai_model,
            "api_key": self.settings.openai_api_key,
            "timeout": 90,
            "max_retries": 2,
        }
        if self.base_url:
            model_kwargs["base_url"] = self.base_url

        return Agent(
            name="MapChat",
            model=OpenAIChat(**model_kwargs),
            system_message=build_system_prompt(context, search_enabled=bool(tools)),
            # History goes between the system prompt and the new user message.
            add_messages=[Message(role=m.role, content=m.content) for m in history] or None,
            tools=tools or None,
            tool_call_limit=4,
            show_tool_calls=False,  # True would splice "Running: search_google(...)" into replies
            telemetry=False,
            monitoring=False,
        )

    def stream(
        self, message: str, history: Sequence[HistoryMessage], context: Sequence[str]
    ) -> Iterator[StreamEvent]:
        from agno.run.response import RunEvent

        agent = self._build_agent(history, context)
        for chunk in agent.run(message, stream=True, stream_intermediate_steps=True):
            event = getattr(chunk, "event", RunEvent.run_response.value)
            if event == RunEvent.run_response.value:
                if isinstance(chunk.content, str) and chunk.content:
                    yield StreamEvent("delta", text=chunk.content)
            elif event == RunEvent.tool_call_started.value:
                yield StreamEvent("status", stage="searching", detail=_search_detail(chunk))
            elif event == RunEvent.tool_call_completed.value:
                yield StreamEvent("status", stage="thinking", detail="Reading the results")


def describe_error(exc: BaseException) -> tuple[str, str]:
    """Map an exception from the model stack to a (code, user-facing message) pair."""
    chain: list[BaseException] = []
    current: BaseException | None = exc
    while current is not None and len(chain) < 8:
        chain.append(current)
        current = current.__cause__ or current.__context__

    try:
        import openai
    except ImportError:  # pragma: no cover - openai is a hard dependency
        openai = None  # type: ignore[assignment]

    for err in chain:
        if openai is not None:
            if isinstance(err, openai.AuthenticationError):
                return "llm_auth", "The OpenAI API key was rejected. Check OPENAI_API_KEY."
            if isinstance(err, openai.RateLimitError):
                return (
                    "llm_rate_limited",
                    "The language model is rate-limited or out of quota. Try again shortly.",
                )
            if isinstance(err, openai.NotFoundError):
                return "llm_model", "The configured OpenAI model is not available. Check OPENAI_MODEL."
            if isinstance(err, (openai.APIConnectionError, openai.APITimeoutError)):
                return "llm_unreachable", "Couldn't reach the language model. Try again in a moment."
        if type(err).__name__ == "ModelRateLimitError":
            return (
                "llm_rate_limited",
                "The language model is rate-limited or out of quota. Try again shortly.",
            )
    return "internal", "Something went wrong while generating a reply. Please try again."


def make_chat_backend(settings: "Settings") -> ChatBackend | None:
    if settings.demo:
        from demo import DemoChatBackend

        return DemoChatBackend()
    if settings.openai_api_key:
        return AgnoChatBackend(settings)
    return None

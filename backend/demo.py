"""Offline demo mode (MAPCHAT_DEMO=1): canned, streamed answers and a static geocoder.

Lets you develop and preview the UI without OpenAI, SerpAPI or Google Maps keys.
Coordinates are approximate and for demonstration only.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import Iterator, Sequence

from agent import HistoryMessage, StreamEvent
from geocoding import GeoResult
from locations import display_name, label_key

# label -> (lat, lon, address)
DEMO_PLACES: dict[str, tuple[float, float, str]] = {
    "Shaniwar Wada, Shivajinagar, Pune": (18.5195, 73.8553, "Shaniwar Peth, Pune, Maharashtra 411030, India"),
    "Shrimant Dagdusheth Halwai Ganpati Temple, Budhwar Peth, Pune": (
        18.5164, 73.8561, "Budhwar Peth, Pune, Maharashtra 411002, India"),
    "Vaishali Restaurant, FC Road, Pune": (18.5212, 73.8410, "Fergusson College Road, Shivajinagar, Pune 411004, India"),
    "Raja Dinkar Kelkar Museum, Shukrawar Peth, Pune": (18.5107, 73.8562, "Shukrawar Peth, Pune, Maharashtra 411002, India"),
    "Aga Khan Palace, Kalyani Nagar, Pune": (18.5524, 73.9015, "Samrat Ashok Road, Kalyani Nagar, Pune 411006, India"),
    "Koregaon Park, Pune": (18.5362, 73.8940, "Koregaon Park, Pune, Maharashtra, India"),
    "Osho Teerth Park, Koregaon Park, Pune": (18.5389, 73.8869, "North Main Road, Koregaon Park, Pune 411001, India"),
    "Kayani Bakery, East Street, Camp, Pune": (18.5145, 73.8795, "East Street, Camp, Pune, Maharashtra 411001, India"),
    "Ruby Hall Clinic, Sassoon Road, Pune": (18.5323, 73.8768, "Sassoon Road, Sangamvadi, Pune 411001, India"),
    "Jehangir Hospital, Sassoon Road, Pune": (18.5302, 73.8755, "Sassoon Road, Pune, Maharashtra 411001, India"),
    "Aditya Birla Memorial Hospital, Thergaon, Pimpri-Chinchwad": (
        18.6236, 73.7764, "Thergaon, Pimpri-Chinchwad, Maharashtra 411033, India"),
    "Sassoon General Hospital, Station Road, Pune": (18.5262, 73.8720, "Station Road, Pune, Maharashtra 411001, India"),
    "Third Wave Coffee, Koregaon Park, Pune": (18.5402, 73.8925, "Lane 6, Koregaon Park, Pune 411001, India"),
    "Blue Tokai Coffee Roasters, Kalyani Nagar, Pune": (18.5475, 73.9030, "Kalyani Nagar, Pune, Maharashtra 411006, India"),
    "Pagdandi Books Chai Cafe, Baner, Pune": (18.5605, 73.7865, "Baner, Pune, Maharashtra 411045, India"),
    "Vohuman Cafe, Sassoon Road, Pune": (18.5312, 73.8772, "Sassoon Road, Pune, Maharashtra 411001, India"),
    "Musée d'Orsay, Rue de Lille, Paris": (48.8600, 2.3266, "1 Rue de la Légion d'Honneur, 75007 Paris, France"),
    "Jardin du Luxembourg, Paris": (48.8462, 2.3372, "75006 Paris, France"),
    "Café de Flore, Saint-Germain-des-Prés, Paris": (48.8541, 2.3326, "172 Boulevard Saint-Germain, 75006 Paris, France"),
    "Sainte-Chapelle, Île de la Cité, Paris": (48.8554, 2.3450, "10 Boulevard du Palais, 75001 Paris, France"),
    "Le Marais, Paris": (48.8590, 2.3620, "Le Marais, 75003–75004 Paris, France"),
    "Canal Saint-Martin, Paris": (48.8710, 2.3650, "Quai de Valmy, 75010 Paris, France"),
    "Sinhagad Fort, Pune": (18.3664, 73.7559, "Sinhagad, Pune district, Maharashtra 411025, India"),
    "Rajmachi Fort, Lonavala": (18.8283, 73.3998, "Rajmachi, near Lonavala, Maharashtra 410401, India"),
    "Tamhini Ghat, Maharashtra": (18.4547, 73.4200, "Tamhini Ghat, Mulshi, Maharashtra, India"),
    "Vetal Tekdi, Pune": (18.5236, 73.8235, "Kothrud, Pune, Maharashtra 411038, India"),
    "Parvati Hill, Pune": (18.4970, 73.8466, "Parvati Paytha, Pune, Maharashtra 411009, India"),
}

DEMO_AREAS: list[tuple[float, float, str]] = [
    (18.5204, 73.8567, "Pune, Maharashtra, India"),
    (18.5308, 73.8475, "Shivajinagar, Pune, Maharashtra, India"),
    (18.5362, 73.8940, "Koregaon Park, Pune, Maharashtra, India"),
    (18.5590, 73.7868, "Baner, Pune, Maharashtra, India"),
    (18.6298, 73.7997, "Pimpri-Chinchwad, Maharashtra, India"),
    (48.8566, 2.3522, "Paris, Île-de-France, France"),
]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


class DemoGeocoder:
    name = "demo"

    def __init__(self) -> None:
        self._by_key = {label_key(k): v for k, v in DEMO_PLACES.items()}
        self._by_name = {label_key(display_name(k)): v for k, v in DEMO_PLACES.items()}

    def geocode(self, query: str) -> GeoResult | None:
        hit = self._by_key.get(label_key(query)) or self._by_name.get(label_key(display_name(query)))
        if hit is None:
            return None
        lat, lon, address = hit
        return GeoResult(lat=lat, lon=lon, address=address)

    def reverse(self, lat: float, lon: float) -> str | None:
        dist, name = min((_haversine_km(lat, lon, a_lat, a_lon), n) for a_lat, a_lon, n in DEMO_AREAS)
        return name if dist <= 40 else None


@dataclass(frozen=True)
class Scenario:
    keywords: tuple[str, ...]
    text: str
    search: str | None = None


PUNE_DAY = Scenario(
    keywords=(),
    text="""\
Here's an unhurried day in Pune that mixes Peshwa-era history, old-city food and a leafy \
evening — every stop is within a 20-minute ride of the next.

### Morning — the old city
- **7:30** Start at [LOCATION]Shaniwar Wada, Shivajinagar, Pune[/LOCATION], the 18th-century \
seat of the Peshwas. Early light on the fortified gates is the best part, and it's quiet before \
the tour groups arrive.
- **8:30** Walk ten minutes to [LOCATION]Shrimant Dagdusheth Halwai Ganpati Temple, Budhwar Peth, \
Pune[/LOCATION] — gilded, busy and very much a working temple.
- **9:30** Breakfast at [LOCATION]Vaishali Restaurant, FC Road, Pune[/LOCATION]: order the SPDP \
and a filter coffee. Expect a short wait at weekends.

### Afternoon — museums and shade
- **11:30** The [LOCATION]Raja Dinkar Kelkar Museum, Shukrawar Peth, Pune[/LOCATION] is one man's \
collection of more than 20,000 everyday objects — lamps, nutcrackers, instruments. Allow 90 minutes.
- **14:00** Head east to the [LOCATION]Aga Khan Palace, Kalyani Nagar, Pune[/LOCATION], where \
Gandhi was interned in 1942. The gardens stay calm in the afternoon heat.

### Evening — Koregaon Park
- **17:30** Wander the lanes of [LOCATION]Koregaon Park, Pune[/LOCATION] and take a shaded walk \
in [LOCATION]Osho Teerth Park, Koregaon Park, Pune[/LOCATION].
- **19:30** Finish with Irani chai and Shrewsbury biscuits at [LOCATION]Kayani Bakery, East Street, \
Camp, Pune[/LOCATION] — popular items sell out by evening.

**Getting around:** auto-rickshaws are easiest; budget ₹150–250 per hop. Opening hours vary, \
so check timings before you set out.""",
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        keywords=("hospital", "clinic", "doctor", "emergency", "medical"),
        text="""\
For emergencies, these are the best-equipped multi-specialty hospitals in central and western \
Pune. All of them run 24×7 emergency departments.

1. **[LOCATION]Ruby Hall Clinic, Sassoon Road, Pune[/LOCATION]** — one of the city's oldest \
private hospitals, strong in cardiology and trauma care. Closest if you're near the railway station.
2. **[LOCATION]Jehangir Hospital, Sassoon Road, Pune[/LOCATION]** — a few minutes from Ruby Hall, \
known for critical care and a well-run emergency wing.
3. **[LOCATION]Aditya Birla Memorial Hospital, Thergaon, Pimpri-Chinchwad[/LOCATION]** — a large \
tertiary-care campus serving the western suburbs and the Hinjewadi IT belt.

If cost is a concern, [LOCATION]Sassoon General Hospital, Station Road, Pune[/LOCATION] is the main \
government teaching hospital and treats emergencies regardless of ability to pay.

**In an emergency, call 108** for a state ambulance, or 112 for the national emergency line.""",
    ),
    Scenario(
        keywords=("coffee", "cafe", "café", "laptop", "work from"),
        search="specialty coffee Koregaon Park Kalyani Nagar Baner",
        text="""\
Pune's specialty-coffee scene clusters in Koregaon Park, Kalyani Nagar and Baner. These four are \
good for an unhurried cup — and most of them for an afternoon on a laptop.

- [LOCATION]Third Wave Coffee, Koregaon Park, Pune[/LOCATION] — reliable pour-overs, plenty of \
plug points and a quiet upper floor on weekdays.
- [LOCATION]Blue Tokai Coffee Roasters, Kalyani Nagar, Pune[/LOCATION] — roasts on site; ask for \
the single-estate filter of the week.
- [LOCATION]Pagdandi Books Chai Cafe, Baner, Pune[/LOCATION] — a bookshop café with long tables \
and a no-rush mood; better for chai than espresso.
- [LOCATION]Vohuman Cafe, Sassoon Road, Pune[/LOCATION] — an old-school Irani café. Go early for \
the cheese omelette; it's not a laptop spot.

**Tip:** weekday mornings before 11 am are the quietest. Hours change often, so check before you \
head out.""",
    ),
    Scenario(
        keywords=("paris", "france"),
        search="Paris weekend itinerary museum opening hours",
        text="""\
A first weekend in Paris works best split by river bank: art and the Left Bank on day one, the \
islands and the Marais on day two.

### Day 1 — The Left Bank
- Morning at the [LOCATION]Musée d'Orsay, Rue de Lille, Paris[/LOCATION]. Book a timed ticket and \
go straight to the fifth-floor Impressionists.
- Lunch, then a slow loop of the [LOCATION]Jardin du Luxembourg, Paris[/LOCATION].
- An overpriced but obligatory coffee at [LOCATION]Café de Flore, Saint-Germain-des-Prés, \
Paris[/LOCATION].

### Day 2 — The islands and the Marais
- Start at [LOCATION]Sainte-Chapelle, Île de la Cité, Paris[/LOCATION] on a sunny morning, when \
the stained glass is at its brightest.
- Cross to [LOCATION]Le Marais, Paris[/LOCATION] for galleries, falafel on Rue des Rosiers and \
small boutiques.
- End with a drink along the [LOCATION]Canal Saint-Martin, Paris[/LOCATION].

**Getting around:** the Métro is fastest; a Navigo Easy card saves queuing for paper tickets.""",
    ),
    Scenario(
        keywords=("trek", "hike", "fort", "getaway", "nature", "monsoon", "weekend"),
        text="""\
Within two hours of Pune you can be on a Sahyadri ridge. The monsoon (July–September) is the most \
dramatic season; winter mornings are the most comfortable.

- [LOCATION]Sinhagad Fort, Pune[/LOCATION] — the classic: a 2–3 hour climb, or drive most of the \
way up. Stay for kanda bhaji and matka dahi at the top.
- [LOCATION]Rajmachi Fort, Lonavala[/LOCATION] — a longer, gentle forest trail; camp overnight \
in early June to see the fireflies.
- [LOCATION]Tamhini Ghat, Maharashtra[/LOCATION] — less a trek than a spectacular drive, with \
waterfalls every few kilometres in the rains.

Short on time? Take a sunrise walk up [LOCATION]Vetal Tekdi, Pune[/LOCATION] or \
[LOCATION]Parvati Hill, Pune[/LOCATION] — both are inside the city.

**Go prepared:** carry water and a rain jacket, start early, and keep away from fort edges after \
heavy rain.""",
    ),
)

_TOKEN_RE = re.compile(r"\S+\s*|\s+")


def pick_scenario(message: str) -> Scenario:
    text = message.casefold()
    for scenario in SCENARIOS:
        if any(word in text for word in scenario.keywords):
            return scenario
    return PUNE_DAY


class DemoChatBackend:
    def __init__(self, delay: float = 0.014, pause: float = 1.0) -> None:
        self.delay = delay  # seconds per streamed word
        self.pause = pause  # multiplier for the simulated "thinking"/"searching" pauses

    def _sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)

    def stream(
        self, message: str, history: Sequence[HistoryMessage], context: Sequence[str]
    ) -> Iterator[StreamEvent]:
        scenario = pick_scenario(message)
        self._sleep(0.4 * self.pause)
        if scenario.search:
            yield StreamEvent("status", stage="searching", detail=f"Searching the web for “{scenario.search}”")
            self._sleep(1.1 * self.pause)
            yield StreamEvent("status", stage="thinking", detail="Reading the results")
            self._sleep(0.4 * self.pause)
        for token in _TOKEN_RE.findall(scenario.text):
            yield StreamEvent("delta", text=token)
            self._sleep(self.delay)

import os
import re
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.serpapi import SerpApiTools
from geopy.geocoders import Nominatim
import googlemaps
import time  # Import time module for delay

# Load environment variables
load_dotenv()

# API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_KEY")
google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")  # Add this line

if not google_maps_api_key:
    raise ValueError("GOOGLE_MAPS_API_KEY is missing! Please check your environment variables.")


if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing! Please check your environment variables.")

if not serpapi_api_key:
    raise ValueError("SERPAPI_KEY is missing! Please check your environment variables.")

# Initialize the AI Agent
chatbot = Agent(
    name="Geospatial Query Bot",
    role="An AI assistant for geospatial data retrieval.",
    # Set add_history_to_messages=true to add the previous chat history to the messages sent to the Model.
    add_history_to_messages=True,
    # Number of historical responses to add to the messages.
    num_history_responses=4,
    model=OpenAIChat(id="gpt-4o"),
    tools=[SerpApiTools(api_key=serpapi_api_key)],
    description="An AI that extracts and handles location-based queries.",
    instructions="""You are an advanced geospatial assistant specializing in location intelligence.

LOCATION FORMATTING RULES (HIGHEST PRIORITY):
- ANY geographical entity mentioned in your response MUST be formatted with [LOCATION] tags: [LOCATION]entity name, address details[/LOCATION]
- This includes but is not limited to: cities, neighborhoods, landmarks, businesses, streets, parks, buildings, monuments, natural features, etc.
- Format EACH distinct location separately with its own tags, even in lists or when locations are near each other
- Include precise details when possible: [LOCATION]Starbucks, FC Road, Shivajinagar, Pune[/LOCATION]
- For general areas: [LOCATION]Koregaon Park, Pune[/LOCATION]
- For cities/regions: [LOCATION]Pune, Maharashtra[/LOCATION]

RESPONSE QUALITY GUIDELINES:
- Use **bold formatting** for important information, headings, and key details
- Provide comprehensive, accurate information based on your knowledge
- Engage with both specific and general location queries with equal detail
- Structure complex responses (itineraries, comparisons) logically with clear sections
- For multi-day plans, clearly label days and timeframes
- Balance detail with readability - provide enough context for each recommendation

DATA FRESHNESS:
- Only use SerpAPI for queries requiring real-time data (current events, new businesses, etc.)
- Clearly indicate when information might need verification due to potential changes

EXAMPLES:
"I recommend visiting [LOCATION]Shaniwar Wada, Shivajinagar, Pune[/LOCATION] in the morning, followed by lunch at [LOCATION]Vaishali Restaurant, FC Road, Pune[/LOCATION]."

"The three best hospitals are [LOCATION]Ruby Hall Clinic, Sassoon Road, Pune[/LOCATION], [LOCATION]Jehangir Hospital, Sassoon Road, Pune[/LOCATION], and [LOCATION]Aditya Birla Memorial Hospital, Thergaon, Pimpri-Chinchwad[/LOCATION]."
""",
    show_tool_calls=True,
    markdown=True,
)

# Initialize GeoPy Nominatim Geocoder
gmaps = googlemaps.Client(key=google_maps_api_key)

# Function to extract locations using the enforced format
def extract_locations(response_text: str) -> list:
    """Extract locations enclosed in [LOCATION] ... [/LOCATION]."""
    return list(set(re.findall(r'\[LOCATION\](.*?)\[/LOCATION\]', response_text)))


def convert_to_coordinates(locations: list) -> dict:
    """Convert location names to (latitude, longitude) using Google Maps Geocoding API."""
    coordinates = {}

    for location in locations:
        try:
            # Ensure location format is correct for geocoding
            formatted_location = location.strip()
            
            # Get geocoding result from Google Maps
            geocode_result = gmaps.geocode(formatted_location)
            
            if geocode_result and len(geocode_result) > 0:
                # Extract location data from the first result
                location_data = geocode_result[0]['geometry']['location']
                coordinates[location] = (location_data['lat'], location_data['lng'])
            else:
                coordinates[location] = "Coordinates not found"
                
            # Optional: Add a small delay to respect rate limits if needed
            time.sleep(0.1)
            
        except Exception as e:
            coordinates[location] = f"Error: {e}"

    return coordinates


import os
import re
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.serpapi import SerpApiTools
from geopy.geocoders import Nominatim
import time  # Import time module for delay

# Load environment variables
load_dotenv()

# API keys
groq_api_key = os.getenv("GROQ_API_KEY")
serpapi_api_key = os.getenv("SERPAPI_KEY")



if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing! Please check your environment variables.")

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
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[SerpApiTools(api_key=serpapi_api_key)],
    description="An AI that extracts and handles location-based queries.",
    instructions="""You are a geospatial assistant. 
    - For general queries, answer normally.
    - Use **word** for bold words , remember to use bold where required
    - For location-based queries, **always format locations** inside [LOCATION] ... [/LOCATION].
    - For Each location add name of the city after location name for each corresponding location
    - Example: "The best areas are [LOCATION] Kharadi,Pune [/LOCATION], [LOCATION] Wagholi,Pune [/LOCATION], and [LOCATION] Baner,Pune [/LOCATION]."
    - Only use SerpAPI for real-time or fresh data needs.
    """,
    show_tool_calls=True,
    markdown=True,
)

# Initialize GeoPy Nominatim Geocoder
geolocator = Nominatim(user_agent="geospatial_query_bot")

# Function to extract locations using the enforced format
def extract_locations(response_text: str) -> list:
    """Extract locations enclosed in [LOCATION] ... [/LOCATION]."""
    return list(set(re.findall(r'\[LOCATION\](.*?)\[/LOCATION\]', response_text)))


def convert_to_coordinates(locations: list) -> dict:
    """Convert location names to (latitude, longitude) using GeoPy without including the city."""
    coordinates = {}
    
    for location in locations:
        try:
            # Introduce a delay to avoid hitting Nominatim's rate limit
            time.sleep(1)  
            # Ensure location format is correct for geocoding
            formatted_location = location.strip()  # Remove unnecessary spaces
            geocode_result = geolocator.geocode(location, exactly_one=True, addressdetails=False)
            if geocode_result:
                coordinates[location] = (geocode_result.latitude, geocode_result.longitude)
            else:
                coordinates[location] = "Coordinates not found"
        except Exception as e:
            coordinates[location] = f"Error: {e}"
    
    return coordinates


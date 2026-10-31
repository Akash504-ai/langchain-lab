# tools.py

import os
import requests

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

NIMBLE_API_KEY = os.getenv("NIMBLE_API_KEY")

NIMBLE_SEARCH_URL = "https://sdk.nimbleway.com/v1/search"

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


# ==========================================
# TOOL 1: WEB SEARCH
# ==========================================

@tool
def web_search(query: str) -> str:
    """
    Search the live web for current information, recent news,
    technical documentation, companies, or general knowledge.

    Use this tool when the user asks to search the internet,
    find recent information, or verify a current fact.
    """

    if not NIMBLE_API_KEY:
        return (
            "Web search is unavailable because NIMBLE_API_KEY "
            "is missing from the .env file."
        )

    try:
        response = requests.post(
            NIMBLE_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {NIMBLE_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "focus": "general",
                "search_depth": "fast",
                "max_results": 5,
                "country": "IN",
                "locale": "en",
                "output_format": "markdown"
            },
            timeout=45
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])

        if not results:
            return "No relevant search results were found."

        formatted_results = []

        for index, result in enumerate(results, start=1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            description = result.get("description", "")
            content = result.get("content", "")

            # Prefer the richer content when available.
            snippet = content or description

            # Avoid returning unnecessarily large search results.
            snippet = snippet[:1500]

            formatted_results.append(
                f"{index}. {title}\n"
                f"URL: {url}\n"
                f"Information: {snippet}"
            )

        return "\n\n".join(formatted_results)

    except requests.exceptions.HTTPError as error:
        status_code = (
            error.response.status_code
            if error.response is not None
            else "unknown"
        )

        return (
            f"Web search failed with HTTP status {status_code}. "
            "Check your Nimble API key, account quota, and "
            "request parameters."
        )

    except requests.exceptions.RequestException as error:
        return f"Could not connect to the web search API: {error}"

    except ValueError:
        return "The web search API returned an invalid response."


# ==========================================
# TOOL 2: REAL-TIME WEATHER
# ==========================================

@tool
def get_current_weather(city: str) -> str:
    """
    Get current weather conditions for a city.

    Input should be a city name, such as Kolkata, London,
    New York, or Tokyo.

    Returns current temperature, feels-like temperature,
    humidity, precipitation, wind speed, and weather conditions.
    """

    try:
        # Step 1: Find the city's coordinates.
        geo_response = requests.get(
            GEOCODING_URL,
            params={
                "name": city,
                "count": 5,
                "language": "en",
                "format": "json"
            },
            timeout=15
        )

        geo_response.raise_for_status()

        geo_data = geo_response.json()
        locations = geo_data.get("results", [])

        if not locations:
            return f"Could not find a location matching '{city}'."

        # Prefer the first geocoding match.
        location = locations[0]

        latitude = location["latitude"]
        longitude = location["longitude"]

        city_name = location.get("name", city)
        country = location.get("country", "Unknown")

        # Step 2: Get current weather for the coordinates.
        weather_response = requests.get(
            WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,"
                    "relative_humidity_2m,"
                    "apparent_temperature,"
                    "is_day,"
                    "precipitation,"
                    "weather_code,"
                    "wind_speed_10m"
                ),
                "timezone": "auto"
            },
            timeout=15
        )

        weather_response.raise_for_status()

        weather_data = weather_response.json()
        current = weather_data.get("current", {})
        units = weather_data.get("current_units", {})

        if not current:
            return "The weather service returned no current conditions."

        # WMO weather interpretation codes.
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snowfall",
            73: "Moderate snowfall",
            75: "Heavy snowfall",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with slight hail",
            99: "Thunderstorm with heavy hail"
        }

        weather_code = current.get("weather_code")
        condition = weather_codes.get(
            weather_code,
            "Unknown conditions"
        )

        temperature = current.get("temperature_2m")
        feels_like = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        precipitation = current.get("precipitation")
        wind_speed = current.get("wind_speed_10m")

        temperature_unit = units.get("temperature_2m", "°C")
        wind_unit = units.get("wind_speed_10m", "km/h")
        precipitation_unit = units.get("precipitation", "mm")

        observation_time = current.get("time", "Unknown")
        timezone = weather_data.get("timezone", "Unknown")

        return (
            f"Current weather for {city_name}, {country}\n"
            f"Condition: {condition}\n"
            f"Temperature: {temperature} {temperature_unit}\n"
            f"Feels like: {feels_like} {temperature_unit}\n"
            f"Humidity: {humidity}%\n"
            f"Precipitation: {precipitation} "
            f"{precipitation_unit}\n"
            f"Wind speed: {wind_speed} {wind_unit}\n"
            f"Observation time: {observation_time}\n"
            f"Timezone: {timezone}\n"
            "Source: Open-Meteo"
        )

    except requests.exceptions.RequestException as error:
        return f"Could not retrieve weather information: {error}"

    except (ValueError, KeyError, TypeError):
        return (
            "The weather service returned unexpected data. "
            "Please try again."
        )


# ==========================================
# AVAILABLE TOOLS
# ==========================================

tools = [
    web_search,
    get_current_weather
]
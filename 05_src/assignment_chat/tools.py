"""Tools for the chat assistant: Weather API, Music Search, Calculator."""

import os
import json
import requests
from langchain.tools import tool

# ============================================================
# SERVICE 1: Weather API (using wttr.in - free, no API key)
# ============================================================

@tool
def get_weather(city: str) -> str:
    """Get current weather information for a city.
    
    Args:
        city: Name of the city to get weather for (e.g., "Toronto", "London", "Tokyo")
    
    Returns:
        A natural language description of current weather conditions.
    """
    try:
        # wttr.in is a free weather service that returns JSON
        url = f"https://wttr.in/{city}?format=j1"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Extract relevant weather information
        current = data.get("current_condition", [{}])[0]
        location = data.get("nearest_area", [{}])[0]
        
        # Get location details
        area_name = location.get("areaName", [{}])[0].get("value", city)
        country = location.get("country", [{}])[0].get("value", "")
        
        # Get weather details
        temp_c = current.get("temp_C", "N/A")
        feels_like = current.get("FeelsLikeC", "N/A")
        humidity = current.get("humidity", "N/A")
        weather_desc = current.get("weatherDesc", [{}])[0].get("value", "Unknown")
        wind_speed = current.get("windspeedKmph", "N/A")
        wind_dir = current.get("winddir16Point", "")
        
        # Transform to natural language (not verbatim API output)
        weather_report = (
            f"Current weather in {area_name}, {country}: "
            f"{weather_desc} with a temperature of {temp_c}°C "
            f"(feels like {feels_like}°C). "
            f"Humidity is at {humidity}% with winds from the {wind_dir} "
            f"at {wind_speed} km/h."
        )
        
        return weather_report
        
    except requests.exceptions.RequestException as e:
        return f"Unable to fetch weather for {city}. Please check the city name and try again."
    except (KeyError, IndexError, json.JSONDecodeError):
        return f"Unable to parse weather data for {city}. Please try a different city."


# ============================================================
# SERVICE 2: Music Semantic Search (ChromaDB with Pitchfork reviews)
# ============================================================

import chromadb
from openai import OpenAI

# Path to persistent ChromaDB storage
CHROMA_PATH = os.path.join(os.path.dirname(__file__), "chroma_data")


def get_query_embedding(query: str) -> list[float]:
    """Get embedding for a search query."""
    client = OpenAI(
        base_url="https://k7uffyg03f.execute-api.us-east-1.amazonaws.com/prod/openai/v1",
        api_key="any",
        default_headers={"x-api-key": os.getenv("API_GATEWAY_KEY")}
    )
    
    response = client.embeddings.create(
        input=[query],
        model="text-embedding-3-small"
    )
    
    return response.data[0].embedding


def get_music_collection():
    """Get the ChromaDB collection for music reviews."""
    # Persistent client stores data to disk
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        collection = client.get_collection(name="music_reviews")
    except Exception:
        # Collection doesn't exist - need to run embedding script first
        collection = None
    
    return collection

@tool
def search_music_reviews(query: str, num_results: int = 3) -> str:
    """Search Pitchfork music reviews using semantic search.
    
    Args:
        query: The search query about music, albums, or artists
        num_results: Number of results to return (default 3, max 5)
    
    Returns:
        Relevant music reviews matching the query.
    """
    collection = get_music_collection()
    
    if collection is None:
        return "Music database not initialized. Please run the embedding script first."
    
    # Limit results
    num_results = min(max(1, num_results), 5)
    
    try:
        # Get query embedding
        query_embedding = get_query_embedding(query)
        
        # Search using embedding
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=num_results
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "No matching music reviews found for your query."
        
        # Format results as natural language
        formatted_results = []
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            artist = metadata.get("artist", "Unknown Artist")
            title = metadata.get("title", "Unknown Album")
            score = metadata.get("score", "N/A")
            
            formatted_results.append(
                f"**{artist} - {title}** (Score: {score}/10)\n{doc}"
            )
        
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        return f"Error searching music reviews: {str(e)}"


# ============================================================
# SERVICE 3: Calculator (Function Calling)
# ============================================================

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: A mathematical expression to evaluate (e.g., "2 + 2", "sqrt(16)", "15 * 3.14")
    
    Returns:
        The result of the calculation.
    """
    import math
    
    # Safe math functions to allow
    safe_functions = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "abs": abs,
        "round": round,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }
    
    try:
        # Clean the expression
        expr = expression.strip()
        
        # Evaluate safely with only math functions
        result = eval(expr, {"__builtins__": {}}, safe_functions)
        
        # Format result nicely
        if isinstance(result, float):
            if result.is_integer():
                return str(int(result))
            return f"{result:.6g}"
        return str(result)
        
    except ZeroDivisionError:
        return "Error: Division by zero"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


@tool
def unit_convert(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between common units.
    
    Args:
        value: The numeric value to convert
        from_unit: The source unit (e.g., "km", "miles", "celsius", "fahrenheit", "kg", "lb")
        to_unit: The target unit
    
    Returns:
        The converted value with units.
    """
    conversions = {
        # Distance
        ("km", "miles"): lambda x: x * 0.621371,
        ("miles", "km"): lambda x: x * 1.60934,
        ("m", "ft"): lambda x: x * 3.28084,
        ("ft", "m"): lambda x: x * 0.3048,
        ("cm", "in"): lambda x: x * 0.393701,
        ("in", "cm"): lambda x: x * 2.54,
        
        # Temperature
        ("celsius", "fahrenheit"): lambda x: (x * 9/5) + 32,
        ("fahrenheit", "celsius"): lambda x: (x - 32) * 5/9,
        ("c", "f"): lambda x: (x * 9/5) + 32,
        ("f", "c"): lambda x: (x - 32) * 5/9,
        
        # Weight
        ("kg", "lb"): lambda x: x * 2.20462,
        ("lb", "kg"): lambda x: x * 0.453592,
        ("g", "oz"): lambda x: x * 0.035274,
        ("oz", "g"): lambda x: x * 28.3495,
        
        # Volume
        ("l", "gal"): lambda x: x * 0.264172,
        ("gal", "l"): lambda x: x * 3.78541,
        ("ml", "oz"): lambda x: x * 0.033814,
        ("oz", "ml"): lambda x: x * 29.5735,
    }
    
    # Normalize units to lowercase
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()
    
    key = (from_unit, to_unit)
    
    if key in conversions:
        result = conversions[key](value)
        return f"{value} {from_unit} = {result:.4g} {to_unit}"
    else:
        return f"Conversion from {from_unit} to {to_unit} is not supported. Supported units: km/miles, m/ft, celsius/fahrenheit, kg/lb, l/gal"


# List of all tools for the agent
ALL_TOOLS = [get_weather, search_music_reviews, calculate, unit_convert]

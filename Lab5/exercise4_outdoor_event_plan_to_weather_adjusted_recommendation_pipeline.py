import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)

class EventDetails(BaseModel):
    event_type: str = Field(description="Type of event")
    event_date: str = Field(description="Date of event")
    location: str = Field(description="Event location")
    indoor_outdoor_preference: str = Field(description="Indoor or outdoor preference")


step1_parser = PydanticOutputParser(pydantic_object=EventDetails)


step1_prompt = PromptTemplate(
    template="""
    Analyze the following event description.

    Extract:
    - event type
    - event date
    - location
    - indoor/outdoor preference

    Event Description:
    {event_description_text}

    Return ONLY JSON:
    {{
        "event_type": "...",
        "event_date": "...",
        "location": "...",
        "indoor_outdoor_preference": "..."
    }}
    """,
    input=["event_description_text"]
)


step1_chain = step1_prompt | llm | step1_parser


class WeatherData(BaseModel):
    temperature: str
    precipitation_chance: str
    conditions: str


def get_weather(location, event_date):
    api_key = os.getenv("WEATHER_API_KEY")
    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {"key": api_key,"q": location,"dt": event_date}
    response = requests.get(url, params=params)
    data = response.json()
    weather = WeatherData(temperature=str(data["forecast"]["forecastday"][0]["day"]["avgtemp_c"]) + " °C",
        precipitation_chance=str(data["forecast"]["forecastday"][0]["day"]["daily_chance_of_rain"]) + "%",
        conditions=data["forecast"]["forecastday"][0]["day"]["condition"]["text"]
    )
    return weather

step3_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an event planning assistant.

        Based on the event details and weather data,
        recommend whether to:
        - proceed as planned
        - reschedule
        - move indoors

        Give a short and clear recommendation.
        """
    ),
    (
        "user",
        """
        Event Type: {event_type}
        Event Date: {event_date}
        Location: {location}
        Preference: {indoor_outdoor_preference}

        Temperature: {temperature}
        Precipitation Chance: {precipitation_chance}
        Conditions: {conditions}

        Give the final recommendation.
        """
    )
])


step3_parser = StrOutputParser()

step3_chain = step3_prompt | llm | step3_parser


event_description_text = input("Enter event description: ")

step1_result = step1_chain.invoke({"event_description_text": event_description_text})

print("\nSTEP 1: Event Details")
print(step1_result)


step2_result = get_weather(step1_result.location,step1_result.event_date)

print("\nSTEP 2: Weather Data")
print(step2_result)


step3_result = step3_chain.invoke({
    "event_type": step1_result.event_type,
    "event_date": step1_result.event_date,
    "location": step1_result.location,
    "indoor_outdoor_preference": step1_result.indoor_outdoor_preference,

    "temperature": step2_result.temperature,
    "precipitation_chance": step2_result.precipitation_chance,
    "conditions": step2_result.conditions
})

print("\nSTEP 3: Recommendation")
print(step3_result)
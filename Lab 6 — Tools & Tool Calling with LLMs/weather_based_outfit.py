from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

@tool
def get_weather(city: str) -> str:
    """Return sample current weather temperature and conditions for a city."""
    sample_weather = {
        "Mumbai": {"temperature": 30, "conditions": "humid and partly cloudy"},
        "Pune": {"temperature": 25, "conditions": "partly cloudy"},
        "Delhi": {"temperature": 22, "conditions": "cool and clear"}
    }
    data = sample_weather.get(city.title())
    if not data:
        return f"Weather data unavailable for {city}. Do not guess the weather."
    return str(data)

@tool
def recommend_outfit(temperature: float, conditions: str) -> str:
    """Recommend an outfit based on temperature in Celsius and weather conditions."""
    if temperature >= 30:
        outfit = "light breathable clothing"
    elif temperature <= 18:
        outfit = "a warm layer or jacket"
    else:
        outfit = "comfortable layered clothing"
    if "rain" in conditions.lower():
        outfit += " and bring an umbrella or raincoat"
    return f"Suggestion: wear {outfit}. Conditions: {conditions}."

tools = [get_weather, recommend_outfit]
llm_with_tools = llm.bind_tools(tools)
user_query = "What should I wear today in Mumbai?"
messages = [{"role": "user", "content": user_query}]

first_message = llm_with_tools.invoke(messages)
print("Round 1 - model tool_calls:")
print(first_message.tool_calls)
messages.append(first_message)

weather_calls = [c for c in first_message.tool_calls if c["name"] == "get_weather"]
weather_result = None
for call in weather_calls:
    weather_result = get_weather.invoke(call["args"])
    print("\nFetched weather data:")
    print(weather_result)
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(weather_result)})

# Give the model the real tool result, then let it request the outfit tool.
second_message = llm_with_tools.invoke(messages)
print("\nRound 2 - model tool_calls:")
print(second_message.tool_calls)
messages.append(second_message)

for call in second_message.tool_calls:
    if call["name"] == "recommend_outfit":
        result = recommend_outfit.invoke(call["args"])
        print("\nOutfit tool result:")
        print(result)
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

final_answer = llm_with_tools.invoke(messages)
print("\nFinal recommendation:")
print(final_answer.content)

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
def get_flight_cost(origin: str, destination: str) -> str:
    """Return a sample estimated round-trip flight cost for a route in INR."""
    sample = {("Delhi", "Goa"): 8500, ("Mumbai", "Goa"): 4500, ("Pune", "Delhi"): 7000}
    cost = sample.get((origin.title(), destination.title()))
    if cost is None:
        return f"Sample flight estimate unavailable for {origin} to {destination}."
    return f"Estimated round-trip flight cost: ₹{cost}."

@tool
def get_hotel_cost(city: str, nights: int) -> str:
    """Return a sample hotel cost estimate in INR for the requested number of nights."""
    if nights <= 0:
        return "Nights must be positive."
    nightly_rate = {"Goa": 3000, "Delhi": 3500, "Mumbai": 4000}.get(city.title())
    if nightly_rate is None:
        return f"Sample hotel estimate unavailable for {city}."
    return f"Estimated hotel cost for {nights} night(s): ₹{nightly_rate * nights}."

@tool
def get_food_budget(city: str, days: int) -> str:
    """Return a sample food budget estimate in INR for the requested number of days."""
    if days <= 0:
        return "Days must be positive."
    daily_budget = {"Goa": 1000, "Delhi": 1200, "Mumbai": 1300}.get(city.title())
    if daily_budget is None:
        return f"Sample food estimate unavailable for {city}."
    return f"Estimated food budget for {days} day(s): ₹{daily_budget * days}."

tools = [get_flight_cost, get_hotel_cost, get_food_budget]
llm_with_tools = llm.bind_tools(tools)
user_query = "I'm planning a 3-night trip from Delhi to Goa. What will flights, hotel, and food roughly cost?"
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print("Model suggested tool_calls:")
print(ai_message.tool_calls)

tool_messages = []
for call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == call["name"])
    result = selected_tool.invoke(call["args"])
    print(f"\nResult from {call['name']}:")
    print(result)
    tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

messages.append(ai_message)
messages.extend(tool_messages)
final_answer = llm_with_tools.invoke(messages)
print("\nFinal combined answer:")
print(final_answer.content)

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
def find_restaurants(cuisine: str, city: str) -> str:
    """Find sample restaurants matching a cuisine in a city."""
    data = {
        ("Italian", "Pune"): ["Bella Italia", "Little Italy"],
        ("Indian", "Pune"): ["Shabree", "Durvankur"],
        ("Italian", "Mumbai"): ["Example Italian Bistro"]
    }
    return str(data.get((cuisine.title(), city.title()), []))

@tool
def get_restaurant_rating(restaurant_name: str) -> str:
    """Return a sample rating out of 5 for a restaurant."""
    ratings = {"Bella Italia": 4.3, "Little Italy": 4.1, "Shabree": 4.4, "Durvankur": 4.2, "Example Italian Bistro": 4.0}
    rating = ratings.get(restaurant_name)
    return f"Sample rating: {rating}/5" if rating is not None else "Rating unavailable in sample data."

@tool
def check_restaurant_open(restaurant_name: str, day: str) -> str:
    """Check sample opening status for a restaurant on a day."""
    return f"Sample data: {restaurant_name} is listed as open on {day}."

@tool
def estimate_cost_for_two(restaurant_name: str) -> str:
    """Return a sample approximate cost for two people in INR."""
    prices = {"Bella Italia": 1800, "Little Italy": 1600, "Shabree": 900, "Durvankur": 700, "Example Italian Bistro": 2000}
    price = prices.get(restaurant_name)
    return f"Sample cost for two: ₹{price}" if price is not None else "Cost estimate unavailable in sample data."

tools = [find_restaurants, get_restaurant_rating, check_restaurant_open, estimate_cost_for_two]
llm_with_tools = llm.bind_tools(tools)
user_query = "Suggest a good Italian restaurant in Pune that's open today, with its rating and rough cost for two."
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print("Model suggested tool_calls:")
print(ai_message.tool_calls)
messages.append(ai_message)

for call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == call["name"])
    result = selected_tool.invoke(call["args"])
    print(f"\nResult from {call['name']}:")
    print(result)
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

final_answer = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(final_answer.content)

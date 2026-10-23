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

from datetime import datetime

@tool
def check_calendar_availability(person: str, date: str) -> str:
    """Check sample calendar availability for a person on a date."""
    return f"Sample calendar: {person} is available on {date} between 10:00 and 16:00."

@tool
def get_timezone_offset(city: str) -> str:
    """Return a sample timezone offset for a city."""
    zones = {"Mumbai": "UTC+05:30", "Pune": "UTC+05:30", "London": "UTC+00:00", "New York": "UTC-04:00"}
    return zones.get(city.title(), f"Timezone not found for {city}.")

@tool
def book_meeting(person: str, date: str, time: str) -> str:
    """Book a sample meeting for a person, date, and time."""
    return f"Sample booking confirmed: meeting with {person} on {date} at {time}."

tools = [check_calendar_availability, get_timezone_offset, book_meeting]
llm_with_tools = llm.bind_tools(tools)

def run_query(user_query):
    messages = [{"role": "user", "content": user_query}]
    ai_message = llm_with_tools.invoke(messages)
    print("\nUser query:", user_query)
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

run_query("Is Alex available for a meeting on 2026-11-02?")
run_query("Schedule a meeting with Alex in Mumbai on 2026-11-02 at 14:00. Check availability and timezone first.")

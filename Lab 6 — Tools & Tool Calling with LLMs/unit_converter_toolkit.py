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
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit. Use units C or F."""
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    if from_unit == "C" and to_unit == "F":
        result = value * 9 / 5 + 32
    elif from_unit == "F" and to_unit == "C":
        result = (value - 32) * 5 / 9
    else:
        return "Unsupported temperature conversion. Use C to F or F to C."
    return f"{value}°{from_unit} = {result:.2f}°{to_unit}"

@tool
def convert_distance(value: float, from_unit: str, to_unit: str) -> str:
    """Convert distance between kilometers and miles. Use units km or miles."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit == "km" and to_unit == "miles":
        result = value * 0.621371
    elif from_unit == "miles" and to_unit == "km":
        result = value / 0.621371
    else:
        return "Unsupported distance conversion. Use km to miles or miles to km."
    return f"{value} {from_unit} = {result:.2f} {to_unit}"

tools = [convert_temperature, convert_distance]
llm_with_tools = llm.bind_tools(tools)

user_query = "Convert 100 km to miles."
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print("Model suggested tool_calls:")
print(ai_message.tool_calls)

tool_messages = []
for call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == call["name"])
    result = selected_tool.invoke(call["args"])
    print("\nManually executed result:")
    print(result)
    tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

messages.append(ai_message)
messages.extend(tool_messages)
final_answer = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(final_answer.content)

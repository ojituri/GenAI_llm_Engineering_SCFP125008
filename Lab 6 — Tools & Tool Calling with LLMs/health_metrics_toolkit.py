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

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class BMIInput(BaseModel):
    weight_kg: float = Field(gt=0, description="Weight in kilograms")
    height_cm: float = Field(gt=0, description="Height in centimeters")

class CaloriesInput(BaseModel):
    weight_kg: float = Field(gt=0)
    height_cm: float = Field(gt=0)
    age: int = Field(gt=0)
    activity_level: str = Field(description="One of sedentary, light, moderate, active")

def calculate_bmi(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI from weight in kg and height in cm."""
    height_m = height_cm / 100
    bmi = weight_kg / (height_m * height_m)
    return f"BMI: {bmi:.1f}"

def calculate_daily_calories(weight_kg: float, height_cm: float, age: int, activity_level: str) -> str:
    """Estimate daily calories using a simplified formula. This estimate does not account for sex or individual medical factors."""
    # Simplified Mifflin-St Jeor estimate using an average offset; for learning purposes only.
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age - 78
    multipliers = {"sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725}
    multiplier = multipliers.get(activity_level.lower())
    if multiplier is None:
        return "Unknown activity level. Use sedentary, light, moderate, or active."
    return f"Approximate daily calorie estimate: {base * multiplier:.0f} kcal/day (rough educational estimate)."

bmi_tool = StructuredTool.from_function(
    func=calculate_bmi,
    name="calculate_bmi",
    description="Calculate BMI from weight_kg and height_cm.",
    args_schema=BMIInput
)
calorie_tool = StructuredTool.from_function(
    func=calculate_daily_calories,
    name="calculate_daily_calories",
    description="Estimate daily calorie needs from weight_kg, height_cm, age, and activity_level.",
    args_schema=CaloriesInput
)

class HealthToolkit:
    def get_tools(self):
        return [bmi_tool, calorie_tool]

toolkit = HealthToolkit()
tools = toolkit.get_tools()
print("Toolkit tools:")
print([t.name for t in tools])

llm_with_tools = llm.bind_tools(tools)
user_query = "I'm 70kg, 175cm tall, 25 years old, moderately active. What's my BMI and daily calorie need?"
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print("\nModel suggested tool_calls:")
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
print("\nFinal answer:")
print(final_answer.content)

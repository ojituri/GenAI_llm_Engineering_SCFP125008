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

class EMIInput(BaseModel):
    principal: float = Field(gt=0, description="Loan principal amount")
    annual_rate: float = Field(ge=0, le=30, description="Annual interest rate percentage from 0 to 30")
    tenure_months: int = Field(gt=0, description="Loan tenure in months")

def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    """Calculate monthly EMI using principal, annual interest rate, and tenure in months."""
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    return f"Monthly EMI: ₹{emi:.2f}"

emi_tool = StructuredTool.from_function(
    func=calculate_emi,
    name="calculate_emi",
    description="Calculate monthly loan EMI. Inputs: principal, annual_rate, tenure_months.",
    args_schema=EMIInput
)
tools = [emi_tool]
llm_with_tools = llm.bind_tools(tools)

user_query = "What's my EMI for a 5 lakh loan at 9% for 24 months?"
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)

print("Model suggested tool_calls:")
print(ai_message.tool_calls)

tool_messages = []
for call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == call["name"])
    result = selected_tool.invoke(call["args"])
    print("\nManually executed tool result:")
    print(result)
    tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

messages.append(ai_message)
messages.extend(tool_messages)
final_answer = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(final_answer.content)

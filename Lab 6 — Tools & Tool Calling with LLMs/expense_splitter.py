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

class ExpenseInput(BaseModel):
    total_amount: float = Field(gt=0, description="Total bill amount")
    num_people: int = Field(gt=0, description="Number of people sharing the bill")

def split_expense(total_amount: float, num_people: int) -> str:
    """Split a total bill equally among a positive number of people."""
    share = total_amount / num_people
    return f"Each person owes ₹{share:.2f}."

expense_tool = StructuredTool.from_function(
    func=split_expense,
    name="split_expense",
    description="Split a bill equally. Provide total_amount and num_people.",
    args_schema=ExpenseInput
)
tools = [expense_tool]
llm_with_tools = llm.bind_tools(tools)

total_amount = 2400
num_people = 4
manual_result = expense_tool.invoke({"total_amount": total_amount, "num_people": num_people})
print("Manual call result:")
print(manual_result)

user_query = "Split ₹2400 between 4 friends."
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print("\nModel suggested tool_calls:")
print(ai_message.tool_calls)

tool_messages = []
bound_result = None
for call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == call["name"])
    bound_result = selected_tool.invoke(call["args"])
    print("\nManually executed bound-flow result:")
    print(bound_result)
    tool_messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(bound_result)})

messages.append(ai_message)
messages.extend(tool_messages)
final_answer = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(final_answer.content)
print("\nDo manual and bound-flow results match?", manual_result == bound_result)

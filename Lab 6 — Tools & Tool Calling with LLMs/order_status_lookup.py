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
def lookup_order_status(order_id: str) -> str:
    """Look up the shipping status for an order ID."""
    orders = {
        "ORD-4521": "Shipped; expected delivery in 2 days.",
        "ORD-1001": "Processing; not yet shipped.",
        "ORD-7832": "Delivered yesterday."
    }
    return orders.get(order_id, "Order ID not found in the sample database.")

tools = [lookup_order_status]
llm_with_tools = llm.bind_tools(tools)

user_query = "Where is my order ORD-4521?"
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)

print("Model suggested tool_calls:")
print(ai_message.tool_calls)

tool_results = []
for tool_call in ai_message.tool_calls:
    selected_tool = next(t for t in tools if t.name == tool_call["name"])
    result = selected_tool.invoke(tool_call["args"])
    print("\nManually executed tool result:")
    print(result)
    tool_results.append({
        "role": "tool",
        "tool_call_id": tool_call["id"],
        "content": str(result)
    })

messages.append(ai_message)
messages.extend(tool_results)
final_answer = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(final_answer.content)

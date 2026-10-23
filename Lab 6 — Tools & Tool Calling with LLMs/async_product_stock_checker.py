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

import asyncio
from typing import Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

class StockInput(BaseModel):
    product_id: str = Field(description="Product ID to check")

class ProductStockTool(BaseTool):
    name: str = "check_product_stock"
    description: str = "Check whether a product is in stock using its product ID."
    args_schema: Type[BaseModel] = StockInput

    stock = {
        "P1023": "In stock: 12 units available.",
        "P2040": "Out of stock.",
        "P5501": "In stock: 3 units available."
    }

    def _run(self, product_id: str) -> str:
        return self.stock.get(product_id, "Product ID not found.")

    async def _arun(self, product_id: str) -> str:
        # Simulates an asynchronous lookup.
        await asyncio.sleep(0)
        return self._run(product_id)

stock_tool = ProductStockTool()

print("Manual synchronous invoke:")
print(stock_tool.invoke({"product_id": "P1023"}))

async def run_async_example():
    print("\nManual asynchronous ainvoke:")
    print(await stock_tool.ainvoke({"product_id": "P1023"}))

asyncio.run(run_async_example())

llm_with_tools = llm.bind_tools([stock_tool])
user_query = "Is product P1023 in stock?"
messages = [{"role": "user", "content": user_query}]
ai_message = llm_with_tools.invoke(messages)
print("\nModel suggested tool_calls:")
print(ai_message.tool_calls)

for call in ai_message.tool_calls:
    result = stock_tool.invoke(call["args"])
    print("\nTool result:")
    print(result)
    messages.append(ai_message)
    messages.append({"role": "tool", "tool_call_id": call["id"], "content": str(result)})

final_answer = llm_with_tools.invoke(messages)
print("\nFinal answer:")
print(final_answer.content)

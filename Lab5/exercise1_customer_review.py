import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)

class ComplaintData(BaseModel):
    complaint : str = Field(description="The core complaint made by the customer: ")
    product_or_feature : str = Field(description="The product or feature mentioned by the customer: ")
    sentiment : str = Field(description="The customer's sentiment: ")

parser = PydanticOutputParser(pydantic_object=ComplaintData)

review_text= input("Enter Customer review: ")

review_prompt = PromptTemplate.from_template(template="""Analyze the following customer review.

    Extract:
    - core complaint
    - product or feature mentioned
    - customer sentiment

    Customer Review:
    {review_text}

    Return the answer as JSON with these fields:
    {{
        "complaint": "...",
        "product_or_feature": "...",
        "sentiment": "..."
    }}
    """, 
    input = ["review_text"]
)

step1_chain = review_prompt | llm | parser

result = step1_chain.invoke({"review_text": review_text})

print("\nStructured Complaint Data: ",result)

step2_prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a support ticket writer. Create a concise support ticket summary for the internal support team."),
        ("user", """Create a support ticket summary using the following structured information:

        Complaint: {complaint}
        Product or Feature: {product_or_feature}
        Sentiment: {sentiment}
        """)
    ]
)

str_parser = StrOutputParser()

step2_chain = step2_prompt | llm | str_parser

ticket_summary = step2_chain.invoke({"complaint":result.complaint,"product_or_feature":result.product_or_feature,"sentiment":result.sentiment})

print("\nSupport Ticket Summary: ",ticket_summary)
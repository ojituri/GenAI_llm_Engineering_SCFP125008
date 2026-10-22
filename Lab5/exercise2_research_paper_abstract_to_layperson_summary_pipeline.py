import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)

class ResearchData(BaseModel):
    research_question: str = Field(description="The main research question addressed by the paper")
    method:str = Field(description="The research method or approach used in the paper")
    key_finding: str = Field(description="The main finding or result of the research")

parser = PydanticOutputParser(pydantic_object=ResearchData)

step1_prompt = PromptTemplate(template=""" Analyze the following research paper abstract.

    Extract:
    - research question
    - method
    - key finding

    Abstract:
    {abstract_text}

    Return the answer as JSON with these fields:
    {{
        "research_question": "...",
        "method": "...",
        "key_finding": "..."
    }}
    """,
    input=["abstract_text"]
)

step1_chain = step1_prompt | llm | parser

step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a science communicator explaining research findings
        to a general audience.

        Create a clear, concise, and easy-to-understand plain-language
        summary using only the structured research question, method,
        and key finding provided to you.

        Do not assume or add information that is not present in
        the structured data.
        """
    ),
    (
        "user",
        """
        Research Question: {research_question}

        Method: {method}

        Key Finding: {key_finding}

        Write a plain-language summary for a non-expert audience.
        """
    )
])

step2_parser = StrOutputParser()

step2_chain = step2_prompt | llm | step2_parser

abstract_text = input("Enter research paper abstract: ")

result = step1_chain.invoke({"abstract_text":abstract_text})

print("\nStructured Extraction: ",result)

layperson_summary = step2_chain.invoke({"research_question":result.research_question,"method":result.method,"key_finding":result.key_finding})

print("\nLayperson Summary: ",layperson_summary)
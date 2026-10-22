import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)


class SupportSummary(BaseModel):
    issue: str = Field(description="Main issue reported by the customer")
    customer_tone: str = Field(description="Customer's tone")
    resolution_attempted: str = Field(description="Resolution steps already attempted")

step1_parser = PydanticOutputParser(pydantic_object=SupportSummary)


step1_prompt = PromptTemplate(
    template="""
    Analyze the following messy customer support chat log.

    Extract and normalize:
    - the main issue
    - the customer's tone
    - the resolution attempted so far

    Chat Log:
    {chat_log_text}

    Return ONLY JSON:
    {{
        "issue": "...",
        "customer_tone": "...",
        "resolution_attempted": "..."
    }}
    """,
    input=["chat_log_text"]
)


step1_chain = step1_prompt | llm | step1_parser


class EscalationCheck(BaseModel):
    missing_fields: list[str] = Field(description="Information missing for escalation")
    is_ready_to_escalate: bool = Field(description="Whether the issue is ready for escalation")


step2_parser = PydanticOutputParser(pydantic_object=EscalationCheck)


step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an escalation-readiness reviewer.

        Check the structured support summary for information
        needed to escalate the issue.

        Important information may include:
        - account ID
        - error code
        - issue details
        - resolution attempted

        Identify any missing information.

        If important information is missing, add it to
        missing_fields and set is_ready_to_escalate to false.

        If the available information is sufficient,
        set is_ready_to_escalate to true.

        Use only the structured summary provided.

        Return ONLY JSON in this format:

        {{
            "missing_fields": ["..."],
            "is_ready_to_escalate": true
        }}
        """
    ),
    (
        "user",
        """
        Structured Support Summary:

        Issue: {issue}
        Customer Tone: {customer_tone}
        Resolution Attempted: {resolution_attempted}

        Check whether this issue is ready for escalation.
        """
    )
])


step2_chain = step2_prompt | llm | step2_parser


step3_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a support operations lead.

        Create a prioritized escalation report based only
        on the structured support summary and the
        escalation-readiness check.

        Include:
        - priority
        - issue summary
        - customer tone
        - resolution attempted
        - missing information
        - recommended next action

        Do not add information that is not provided.

        Keep the report clear and concise.
        """
    ),
    (
        "user",
        """
        Structured Support Summary:

        Issue: {issue}
        Customer Tone: {customer_tone}
        Resolution Attempted: {resolution_attempted}

        Escalation Check:

        Missing Fields: {missing_fields}
        Ready to Escalate: {is_ready_to_escalate}

        Generate the escalation report.
        """
    )
])


step3_parser = StrOutputParser()

step3_chain = step3_prompt | llm | step3_parser


chat_log_text = input(
    "Enter customer support chat log: "
)


step1_result = step1_chain.invoke({"chat_log_text": chat_log_text})

print("\nSTEP 1: Structured Summary")
print(step1_result)


step2_result = step2_chain.invoke({
    "issue": step1_result.issue,
    "customer_tone": step1_result.customer_tone,
    "resolution_attempted": step1_result.resolution_attempted
})

print("\nSTEP 2: Escalation Readiness")
print(step2_result)


step3_result = step3_chain.invoke({
    "issue": step1_result.issue,
    "customer_tone": step1_result.customer_tone,
    "resolution_attempted": step1_result.resolution_attempted,
    "missing_fields": step2_result.missing_fields,
    "is_ready_to_escalate": step2_result.is_ready_to_escalate
})

print("\nSTEP 3: Escalation Report")
print(step3_result)
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)

class QuestionAnswer(BaseModel):
    question:str=Field(description="The interview question asked")
    answer:str=Field(description="The candidate's answer to the question")


class InterviewAnswers(BaseModel):
    answers: list[QuestionAnswer] = Field(description="List of interview questions and candidate answers")


step1_parser = PydanticOutputParser(pydantic_object=InterviewAnswers)

step1_prompt = PromptTemplate(template="""
    Analyze the following interview transcript.

    Extract the candidate's answer to each interview question.

    Transcript:
    {transcript_text}

    Return the answer as JSON with this structure:
    {{
        "answers": [
            {{
                "question": "...",
                "answer": "..."
            }}
        ]
    }}
    """,
    input = ["transcript_text"])

step1_chain = step1_prompt | llm | step1_parser

class SkillEvaluation(BaseModel):
    skill:str = Field(description="The skill being assessed")
    demonstrated:bool = Field(description="Whether the candidate clearly demonstrated the skill")
    evidence_or_gap:str = Field(description="Evidence supporting the skill or explanation of the gap")

class SkillEvaluations(BaseModel):
    evaluations:list[SkillEvaluation] = Field(description="Evaluation of each skill")

step2_parser = PydanticOutputParser(pydantic_object=SkillEvaluations)

step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a technical interviewer evaluating skill evidence.

        Evaluate the candidate's answers against the given skills.
        For each skill, determine whether it was clearly demonstrated.

        If a skill was not clearly demonstrated, set demonstrated
        to false and explain the gap.

        Use only the candidate's answers as evidence.
        Do not assume that the candidate has a skill unless it is
        clearly supported by the answers.

        Return the result ONLY as JSON in this format:
        {{
            "evaluations": [
                {{
                    "skill": "...",
                    "demonstrated": true,
                    "evidence_or_gap": "..."
                }}
            ]
        }}
        """
    ),
    (
        "user",
        """
        Candidate Answers:
        {answers}

        Skills to Assess:
        {skills_to_assess}

        Evaluate each skill.
        """
    )
])

step2_chain = step2_prompt | llm | step2_parser


class ScorecardRow(BaseModel):
    skill: str = Field(description="The skill being evaluated")
    demonstrated: bool = Field(description="Whether the skill was demonstrated")
    evidence_or_gap: str = Field(description="Evidence or gap for the skill")


class HiringScorecard(BaseModel):
    scorecard: list[ScorecardRow] = Field(description="List of skill evaluation rows")
    overall_recommendation: str = Field(description="Overall hiring recommendation")


step3_parser = PydanticOutputParser(pydantic_object=HiringScorecard)


step3_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a hiring panel coordinator.

        Create a structured hiring scorecard from the skill evaluation.

        Include every skill, whether it was demonstrated, and the
        supporting evidence or gap.

        Provide an overall hiring recommendation based only on
        the skill evaluation.

        Return the result ONLY as JSON in this format:
        {{
            "scorecard": [
                {{
                    "skill": "...",
                    "demonstrated": true,
                    "evidence_or_gap": "..."
                }}
            ],
            "overall_recommendation": "..."
        }}
        """
    ),
    (
        "user",
        """
        Skill Evaluation:
        {skill_evaluation}

        Create the final hiring scorecard.
        """
    )
])


step3_chain = step3_prompt | llm | step3_parser


transcript_text = input("Enter interview transcript: ")

skills_to_assess = input("Enter skills to assess (comma separated): ")


step1_result = step1_chain.invoke({"transcript_text": transcript_text})

print("\nExtracted Answers: ")
print(step1_result)


step2_result = step2_chain.invoke({"answers": step1_result.answers,"skills_to_assess": skills_to_assess})

print("\nSkill Evaluation: ")
print(step2_result)

step3_result = step3_chain.invoke({"skill_evaluation": step2_result.evaluations})

print("\nFinal Scorecard ")
print(step3_result)
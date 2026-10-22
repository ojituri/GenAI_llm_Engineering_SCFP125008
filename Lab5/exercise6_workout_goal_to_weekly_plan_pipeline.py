import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b",api_key=GROQ_API_KEY)


class FitnessGoal(BaseModel):
    goal_type: str = Field(description="Fitness goal of the person")
    fitness_level: str = Field(description="Current fitness level")
    days_available: int = Field(description="Number of workout days available per week")
    equipment_access: str = Field(description="Available workout equipment")


step1_parser = PydanticOutputParser(pydantic_object=FitnessGoal)


step1_prompt = PromptTemplate(
    template="""
    Analyze the following fitness goal description.

    Extract:
    - goal type
    - current fitness level
    - days available per week
    - equipment access

    Fitness Goal Description:
    {fitness_goal_text}

    Days available per week:
    {days_per_week}

    Return ONLY JSON:
    {{
        "goal_type": "...",
        "fitness_level": "...",
        "days_available": 0,
        "equipment_access": "..."
    }}
    """,
    input=["fitness_goal_text", "days_per_week"]
)

step1_chain = step1_prompt | llm | step1_parser


class WorkoutDay(BaseModel):
    day: str = Field(description="Day of the week")
    workout: str = Field(description="Workout planned for the day")
    duration: str = Field(description="Workout duration")
    notes: str = Field(description="Additional workout notes")


class WeeklyWorkoutPlan(BaseModel):
    weekly_plan: list[WorkoutDay] = Field(description="Day-by-day weekly workout plan")


step2_parser = PydanticOutputParser(pydantic_object=WeeklyWorkoutPlan)


step2_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a certified personal trainer.

        Create a structured weekly workout plan using ONLY
        the extracted fitness details provided by the user.

        Do not use or assume information from the original
        fitness goal description.

        Scale the workout plan according to the number of
        available workout days.

        Return ONLY JSON in this format:

        {{
            "weekly_plan": [
                {{
                    "day": "...",
                    "workout": "...",
                    "duration": "...",
                    "notes": "..."
                }}
            ]
        }}
        """
    ),
    (
        "user",
        """
        Goal Type: {goal_type}
        Fitness Level: {fitness_level}
        Days Available: {days_available}
        Equipment Access: {equipment_access}

        Create the weekly workout plan.
        """
    )
])


step2_chain = step2_prompt | llm | step2_parser


fitness_goal_text = input("Enter your fitness goal description: ")

days_per_week = input("Enter number of days available per week: ")


step1_result = step1_chain.invoke({
    "fitness_goal_text": fitness_goal_text,
    "days_per_week": days_per_week
})

print("\nSTEP 1: Extracted Goal Details")
print(step1_result)


step2_result = step2_chain.invoke({
    "goal_type": step1_result.goal_type,
    "fitness_level": step1_result.fitness_level,
    "days_available": step1_result.days_available,
    "equipment_access": step1_result.equipment_access
})

print("\nSTEP 2: Weekly Workout Plan")
print(step2_result)
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Create a simple agent
researcher = Agent(
    role="Research Analyst",
    goal="Find and analyze information",
    backstory="You are an expert at finding and analyzing information.",
    verbose=True,
    llm="gpt-4-turbo",  # Using a generic model name
)

# Create a simple task
research_task = Task(
    description="Research the latest developments in AI and summarize them.",
    agent=researcher,
    expected_output="A brief summary of latest AI developments."
)

# Create a crew with the agent and task
crew = Crew(
    agents=[researcher],
    tasks=[research_task],
    verbose=True
)

# Just print that the crew was created successfully without actually running it
print("CrewAI initialization successful!")
print(f"Crew version: {crew.__class__.__module__}") 
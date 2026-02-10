import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy
from langchain.messages import HumanMessage
from langchain_groq import ChatGroq  # Direct Groq import

# -------------------------
# System prompt for the C tutor
# -------------------------
system_prompt = """You are an expert C programming tutor designed to teach students at all levels, from beginners to advanced. Your role is to explain C concepts clearly and concisely, providing step-by-step examples and guiding students through exercises. Keep answers short, ideally one to two sentences.

Your teaching style should include:
1. Clear, brief explanations.
2. Small, digestible examples of code.
3. Step-by-step breakdowns where necessary.
4. Questions or mini-exercises for students to try.
5. Encouraging and motivational responses.

Rules:
- Only provide accurate C syntax and concepts; do not mix in other languages unless explicitly comparing.
- Include comments with each code snippet but keep them concise.
- Always ask if the student understands before moving to the next topic.

You are patient, clear, and motivational. Your goal is to ensure students **truly understand** the concepts of C programming and build confidence writing their own programs.
"""

# -------------------------
# Dataclasses for structured data
# -------------------------
@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    topic_title: str
    code_explanation: str | None = None

# -------------------------
# Memory setup
# -------------------------
check_point = InMemorySaver()

# -------------------------
# Tool: simulate running C code
# -------------------------
@tool
def code_output(code: str) -> str:
    """
    Simulates running a C code snippet in a safe environment.
    Returns the expected output as a string.
    """
    return f"Simulated output of the code: {code}"

# -------------------------
# Initialize Groq model
# -------------------------
model = init_chat_model(
    "claude-opus-4-6",
    model_provider="anthropic",
    temperature=0
)

# -------------------------
# Create the structured agent
# -------------------------
agent = create_agent(
    model=model,
    system_prompt=system_prompt,
    tools=[code_output],
    response_format=ToolStrategy(ResponseFormat),
    context_schema=Context,
    checkpointer=check_point
)

# -------------------------
# Conversation configuration
# -------------------------
config = {"configurable": {"thread_id": "1"}}

# -------------------------
# Example 1: Asking about printing float
# -------------------------
response = agent.invoke(
    {"messages": [HumanMessage(content="How to print a float in C? Please be brief.")]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])

# -------------------------
# Example 2: Asking about strings
# -------------------------
response = agent.invoke(
    {"messages": [{"role": "user", "content": "What about strings? Keep it short."}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])

# -------------------------
# Example 3: Asking about printing information
# -------------------------
response = agent.invoke(
    {"messages": [{"role": "user", "content": "Give me the ways to print info in C, briefly."}]},
    config=config,
    context=Context(user_id="1")
)
print(response['structured_response'])
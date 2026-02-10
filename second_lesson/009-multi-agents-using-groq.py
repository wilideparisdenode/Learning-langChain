from dotenv import load_dotenv
load_dotenv()

# 🔒 Disable LangSmith completely
import os
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.messages import HumanMessage

# -----------------------------
# Groq model (shared)
# -----------------------------
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # fast + cheap
    temperature=0
)

# -----------------------------
# Tools
# -----------------------------
@tool
def square_root(x: float) -> float:
    """Calculate the square root of a number"""
    return x ** 0.5

@tool
def square(x: float) -> float:
    """Calculate the square of a number"""
    return x ** 2

# -----------------------------
# Subagents
# -----------------------------
subagent_1 = create_agent(
    model=llm,
    tools=[square_root]
)

subagent_2 = create_agent(
    model=llm,
    tools=[square]
)

# -----------------------------
# Tool wrappers calling subagents
# -----------------------------
@tool
def call_subagent_1(x: float) -> str:
    """Call subagent 1 to calculate square root"""
    response = subagent_1.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square root of {x}")]}
    )
    return response["messages"][-1].content


@tool
def call_subagent_2(x: float) -> str:
    """Call subagent 2 to calculate square"""
    response = subagent_2.invoke(
        {"messages": [HumanMessage(content=f"Calculate the square of {x}")]}
    )
    return response["messages"][-1].content

# -----------------------------
# Main agent
# -----------------------------
main_agent = create_agent(
    model=llm,
    tools=[call_subagent_1, call_subagent_2],
    system_prompt=(
        "You are a calculator agent.\n"
    "For every user question, you MUST call exactly one tool.\n"
    "Do NOT explain.\n"
    "Return ONLY the final numeric result."
    ),
)

# -----------------------------
# Run
# -----------------------------
question = "What is the square root of 456?"

response = main_agent.invoke(
    {"messages": [HumanMessage(content=question)]}
)

print(response["messages"][-1].content)

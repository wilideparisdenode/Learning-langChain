from dotenv import load_dotenv
load_dotenv()
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
os.environ["LANGCHAIN_TRACING_V2"] = "false"
# -----------------------------
# Model (Gemini)
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    streaming=True
)

# -----------------------------
# System prompt
# -----------------------------
system_prompt = "You are an investigative journalist."

# -----------------------------
# Prompt template
# -----------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
    ]
)

# -----------------------------
# Agent-like chain
# -----------------------------
agent = prompt | llm

print("\n=========\n")
print("Who really killed JFK?\n")

# ---- invoke (non-streaming)
response = agent.invoke(
    {"question": "Who really killed JFK?"}
)

print(response.content)

print("\n=========\n")
print("Really, who really killed JFK?\n")

# ---- streaming
for chunk in agent.stream(
    {"question": "Really, who really killed JFK?"}
):
    if chunk.content:
        print(chunk.content, end="", flush=True)

print("\n\n=========\n")

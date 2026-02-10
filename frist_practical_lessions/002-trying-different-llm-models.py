import os
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant"   
)

response = llm.invoke(
    "Explain what open-weight LLMs are in simple terms."
)

print(response.content)

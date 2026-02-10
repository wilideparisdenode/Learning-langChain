import os
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env
_ = load_dotenv(find_dotenv())

# Optional: disable LangSmith tracing noise
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_groq import ChatGroq
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
)

# ------------------------------------------------------------------
# Initialize Groq Chat Model
# ------------------------------------------------------------------

chatModel = ChatGroq(
    model="llama-3.1-8b-instant"
)

# ------------------------------------------------------------------
# 1. PromptTemplate (text-style prompting)
# ------------------------------------------------------------------

prompt_template = PromptTemplate.from_template(
    "Tell me a {adjective} story about {topic}."
)

prompt_text = prompt_template.format(
    adjective="curious",
    topic="the Kennedy family"
)

response = chatModel.invoke(prompt_text)

print("Tell me one curious thing about the Kennedy family:")
print(response.content)

print("\n----------\n")

# ------------------------------------------------------------------
# 2. ChatPromptTemplate (multi-message chat prompting)
# ------------------------------------------------------------------

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an {profession} expert on {topic}."),
        ("human", "Hello, Mr. {profession}, can you please answer a question?"),
        ("ai", "Sure!"),
        ("human", "{user_input}"),
    ]
)

messages = chat_template.format_messages(
    profession="Historian",
    topic="The Kennedy family",
    user_input="How many grandchildren had Joseph P. Kennedy?"
)

response = chatModel.invoke(messages)

print("How many grandchildren had Joseph P. Kennedy?:")
print(response.content)

print("\n----------\n")

# ------------------------------------------------------------------
# 3. Few-shot prompting
# ------------------------------------------------------------------

examples = [
    {"input": "hi!", "output": "¡hola!"},
    {"input": "bye!", "output": "¡adiós!"},
]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an English-Spanish translator."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

messages = final_prompt.format_messages(
    input="How are you today?"
)

response = chatModel.invoke(messages)

print("English → Spanish translation:")
print(response.content)

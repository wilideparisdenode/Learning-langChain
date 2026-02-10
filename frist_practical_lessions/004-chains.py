import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

from langchain_groq import ChatGroq


chatModel = ChatGroq( model="llama-3.1-8b-instant" )


from langchain_core.prompts import ChatPromptTemplate

from langchain_core.prompts import FewShotChatMessagePromptTemplate

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

chain = final_prompt | chatModel

response = chain.invoke({"input": "Who was JFK?"})

print("\n----------\n")

print("Translate: Who was JFK?")
print(response.content)

print("\n----------\n")
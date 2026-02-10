from dotenv import load_dotenv, find_dotenv
import os
from openai import OpenAI

_ = load_dotenv(find_dotenv())

client = OpenAI(base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)

from dotenv import load_dotenv
import os
os.environ["LANGCHAIN_TRACING_V2"]="false"
load_dotenv()


from langchain.chat_models import init_chat_model

model = init_chat_model(
    model="claude-sonnet-4-20250514",
     api_key=os.environ["ANTHROPIC_API_KEY"],
    temperature=0.0
)


print("Who killed JFK?\n")

response = model.invoke("Who killed JFK?")
print(response.content)

print("\n=========\n")

print("What is the best coffee shop in San Francisco?\n")

response = model.invoke("What is the best coffee shop in San Francisco?")
print(response.content)

print("\n=========\n")

print("What is the best Spanish restaurant in San Francisco?\n")

for chunk in model.stream("What is the best Spanish restaurant in San Francisco?"):
    print(chunk.text, end="|", flush=True)
    
print("\n=========\n")

print("What is the best Meetup for European Expats in San Francisco?\n")

for chunk in model.stream("What is the best Meetup for European Expats in San Francisco?"):
    print(chunk.text, end="", flush=True)
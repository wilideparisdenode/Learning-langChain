from dotenv import load_dotenv
load_dotenv()

from langchain.messages import HumanMessage
from langchain.agents import create_agent
from langchain_groq import ChatGroq

# 1. Create the LLM
llm = ChatGroq(model="llama-3.1-8b-instant")

# 2. System prompt
system_prompt = "You are a software engineer."

# 3. Create the agent (pass the LLM, not a string)
agent = create_agent(
    model=llm,
    system_prompt=system_prompt
)

# 4. Invoke the agent
response = agent.invoke(
    {"messages": [HumanMessage(content="How can I really master Generative AI?")]}
)

# 5. Correct way to read output
print(response)
print(response["messages"][-1].content)

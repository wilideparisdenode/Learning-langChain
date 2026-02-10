from tavily import TavilyClient
import os
from langchain.agents import create_agent
from typing import Dict, Any
from langchain.chat_models import  init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"]="false"
model=init_chat_model(
    model="claude-opus-4-6",
    model_provider="anthropic",
    
)

tavily_client = TavilyClient()
@tool
def match_prediction(query: str) -> Dict[str, Any]:
 """The will reture the best Match prediction fron online it can ind"""
 return  tavily_client.search(query)

agent=create_agent(
  model=model,
  system_prompt="you are a  match  prediction specialist , and can give  people match scoreline prediction base on currect performance , and form",
  tools=[match_prediction]
)
 
response=agent.invoke( {"messages":HumanMessage(content="man united , has a match today 11/01/26  vs westham, i waht you to give me a sure match prediction , for this match")})
print(response['messages'][-1].content)
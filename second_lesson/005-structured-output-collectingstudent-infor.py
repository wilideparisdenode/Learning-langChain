from dataclasses import dataclass
import os
os.environ["LANGCHAIN_TRACING_V2"]="false"
from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel
from langchain.chat_models import  init_chat_model
from langchain.agents import create_agent
from langchain.messages import HumanMessage, SystemMessage
from langchain.tools import tool,ToolRuntime
from langgraph.checkpoint.memory import MemorySaver



@dataclass
class Context:
    """this is use information , about the current talking user"""
    user_id: str

check_point=MemorySaver()

class StudentInfo(BaseModel):
    """ this will make the ai model , collect student information"""
    name: str
    studentClass: str
    age: str


s = SystemMessage(content="you are teaching staff collecting data , about student , who want to enroll into our program you are task with collect the students name , class , and age")
h = HumanMessage(content="my name is Ndoping wilson , and i am a level three software engineering student , am 22 , and i want to enroll for the masters program in you school")


model=init_chat_model(
    model="claude-opus-4-6",
    model_provider="anthropic",
    
)
config={"configurable":{"user_id":"1", "thread_id":"1"}}
agent=create_agent(
    model=model,
    system_prompt=s,    
    context_schema=Context(user_id="1"),
    response_format=StudentInfo,
    checkpointer=check_point
)
res=agent.invoke(
    {"messages": [s, h]},
    config=config

)

print(res["structured_response"])

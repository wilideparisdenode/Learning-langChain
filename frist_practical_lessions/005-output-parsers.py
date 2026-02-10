import os
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
openai_api_key = os.environ.get("GROQ_API_KEY")
if not openai_api_key:
    print("Warning: OPENAI_API_KEY not found in environment")

from langchain_groq import ChatGroq
llmModel = ChatGroq(model="llama-3.1-8b-instant")

# Initialize chat model (same model used here for clarity)
chatModel = ChatGroq(model="llama-3.1-8b-instant")

from langchain_core.prompts import PromptTemplate

from langchain_core.output_parsers import SimpleJsonOutputParser

json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)

json_parser = SimpleJsonOutputParser()

json_chain = json_prompt | llmModel | json_parser

response = json_chain.invoke({"question": "What is the biggest country?"})

print("What is the biggest country?")
def _print_response(resp):
    # helper to print different response shapes coming from model or parser
    try:
        if hasattr(resp, "content"):
            print(resp.content)
            return
        import json

        if isinstance(resp, (dict, list)):
            print(json.dumps(resp, indent=2, ensure_ascii=False))
            return
        print(resp)
    except Exception:
        print(repr(resp))


_print_response(response)

print("\n----------\n")


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

# Define your desired data structure.
class Joke(BaseModel):
    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")
    
# Set up a parser
parser = JsonOutputParser(pydantic_object=Joke)

# Inject parser instructions into the prompt template.
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# Create a chain with the prompt and the parser
chain = prompt | chatModel | parser

response = chain.invoke({"query": "Tell me a joke."})

print("Tell me a joke in custom format defined by Pydantic:")
_print_response(response)

print("\n----------\n")
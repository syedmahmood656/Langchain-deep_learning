from langchain_openai import OpenAI # this ia an integration package that let lanchain integrate with open ai
from dotenv import load_dotenv # it loads api key from .env folder into current file

load_dotenv() # invoked dotenv function

llm = OpenAI(model='gpt-3.5turbo-instruct')

result = llm.invoke("what is the capital of india") # with invoke methods help we communicate with gpt model

print(result)
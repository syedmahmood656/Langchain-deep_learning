from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

result = model.invoke("what is the capital of india")

print(result)

print(result.content)

# this code wont work as we dont have openai API key 
# but if we had a key we would have seen the differece in a LLM and Chatmodel
#  
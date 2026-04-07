from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

load_dotenv()

model = ChatAnthropic(model='model name')

result = model.invoke("what is the capital of delhi")

print(result)
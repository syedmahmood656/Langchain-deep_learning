from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
)

model = ChatHuggingFace(llm=llm)

class review(TypedDict):
    summery: str
    sentiment: str

structured_model = model.with_structured_output(output_type=review)

result = structured_model.invoke("I love this product! It's amazing and exceeded my expectations.")

print(result)
print(f"Summary: {result.summery}")
print(f"Sentiment: {result.sentiment}")
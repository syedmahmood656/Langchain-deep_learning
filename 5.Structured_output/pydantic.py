# PYDANTIC : is a data validation and settings management library for Python, based on type annotations. It allows you to define data models with type hints and provides powerful validation and parsing capabilities.
# it ensures the data you work with is correct, structured and type-safe, making it easier to handle complex data structures and configurations in your applications.

from dataclasses import field

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
)

model = ChatHuggingFace(llm=llm)

class review(BaseModel):

    key_themes : list[str] = field(description = "The key themes of the review") #changed from list[str] to field[description = "The key themes of the review"]
    summery: str = Field (description = "A brief summary of the review") #changed from str to Annotated[str, "A brief summary of the review"]
    sentiment:str = Field(description= "The sentiment of the review") #changed from str to Annotated[str, "The sentiment of the review"]
    pros: list[str] = Field(description = "The pros of the product or service") #changed from list[str] to Field(description = "The pros of the product or service")
    cons:list[str] = Field(description= "The cons of the product or service") #chsanged from list[str] to Field(description= "The cons of the product or service")

    name : str = Field(description= "The name of the reviewer") #changed from str to Field(description = "The name of the reviewer")


structured_model = model.with_structured_output(output_type=review)

result = structured_model.invoke("I love this product! It's amazing and exceeded my expectations.")

print(result)
print(f"Summary: {result.summery}")
print(f"Sentiment: {result.sentiment}")
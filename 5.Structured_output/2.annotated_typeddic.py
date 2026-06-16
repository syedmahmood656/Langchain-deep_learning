from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from typing import Optional, TypedDict, Annotated

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
)

model = ChatHuggingFace(llm=llm)

class review(TypedDict):

    key_themes : Annotated[str, "The key themes of the review"] #changed from str to Annotated[str, "The key themes of the review"]
    summery: Annotated[str, "A brief summary of the review"] #changed from str to Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[str, "The sentiment of the review"] #changed from str to Annotated[str, "The sentiment of the review"]
    pros: Annotated[Optional[list[str]], "The pros of the product or service"] #changed from Optional[str] to Annotated[Optional[str], "The pros of the product or service"]
    cons: Annotated[Optional[list[str]], "The cons of the product or service"] #changed from Optional[str] to Annotated[Optional[str], "The cons of the product or service"]    


# this defines a structured output type for the chatbot's response, allowing for more organized and meaningful data extraction. The `review` TypedDict specifies that the output will contain a summary and sentiment, both of which are strings with additional descriptive annotations. This structured approach enables better handling of the chatbot's responses, making it easier to process and analyze the information provided by the model.

structured_model = model.with_structured_output(output_type=review)

result = structured_model.invoke("I love this product! It's amazing and exceeded my expectations.")

print(result)
print(f"Summary: {result.summery}")
print(f"Sentiment: {result.sentiment}")
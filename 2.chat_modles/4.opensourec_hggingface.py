import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Explicitly grab the token from the .env
# If your .env says HUGGINGFACEHUB_ACCESS_TOKEN="...", this will get it
token = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Initialize the LLM with the token passed explicitly
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct", # Use SmolLM2 for better reliability
    task="text-generation",
    huggingfacehub_api_token=token, # <--- ADD THIS LINE
    timeout=300
)

model = ChatHuggingFace(llm=llm)

try:
    result = model.invoke("What is the capital of India?")
    print(result.content)
except Exception as e:
    print(f"Error: {e}")
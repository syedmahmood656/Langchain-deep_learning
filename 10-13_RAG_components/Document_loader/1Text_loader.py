import os
import sys
import io

# Force UTF-8 stdout so Windows console (cp1252) can print unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN")
)

model = ChatHuggingFace(llm=llm)

prompt_template = PromptTemplate(
    template="write the summery of the following text:\n{text}\n\nSummary:",
    input_variables=["text"]
)

parser = StrOutputParser()

loader = TextLoader(
    r'C:\Learnings_career_DS\Deep_learning\Langchain\langchain - LLM,modals and embeddings\10-13_RAG_components\Document_loader\doc.txt',
    encoding='utf-8'
)

docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt_template | model | parser

print(chain.invoke({'text':docs[0].page_content}))

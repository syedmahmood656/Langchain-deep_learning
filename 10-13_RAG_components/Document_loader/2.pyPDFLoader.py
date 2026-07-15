from langchain_community.document_loaders import PyPDFLoader #install pypdf
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("10-13_RAG_components\Document_loader\CNN_project.pdf")

docs = loader.load()

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)
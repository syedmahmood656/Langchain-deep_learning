from langchain_huggingface import HuggingFaceEmbeddings

embedings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

docs = [
    'whats the capital of delhi',
    'tell me about you'
    'how about coffie sometime'
]

vector = embedings.embed_documents(docs)

print(str(vector))
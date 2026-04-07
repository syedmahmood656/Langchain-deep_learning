from langchain_huggingface import HuggingFaceEmbeddings

embedings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

text = "what is the reason behind your unmatched wisdom"

vector = embedings.embed_query(text)

print(str(vector))
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# all the text in thsi doc will bw converted into embeddings
docs = [
    'whats the capital of delhi',
    'tell me about you'
    'how about coffie sometime'
]

embeddings = OpenAIEmbeddings(model='model_name', dimensions=32 )

result = embeddings.embed_query(docs)

print(str(result))


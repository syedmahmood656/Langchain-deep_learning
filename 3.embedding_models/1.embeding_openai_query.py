from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings(model='model_name', dimensions=32 )

result = embeddings.embed_query('delhi is the capital of india')

print(str(result))

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template="tell me 5 facts about {country}?",
    input_variables=["country"])

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"country": "india"})

print(result)

# this code is to visualize the chain graph, you can install graphviz or grandalf to visualize the graph
try:
    chain.get_graph().print_ascii()
except ImportError:
    print("Install graphviz or grandalf to visualize graph: pip install grandalf")
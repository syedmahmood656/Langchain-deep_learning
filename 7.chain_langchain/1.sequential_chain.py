from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt1 = PromptTemplate(
    template="tell me 5 facts about {country}?",
    input_variables=["country"])

prompt2 = PromptTemplate(
    template="summarize the following facts about {facts}",
    input_variables=["facts"])

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

# this is the sequential chain, where the output of the first prompt is passed to the second prompt
chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({"country": "india"})

print(result)

# this code is to visualize the chain graph, you can install graphviz or grandalf to visualize the graph
chain.get_graph().print_ascii()
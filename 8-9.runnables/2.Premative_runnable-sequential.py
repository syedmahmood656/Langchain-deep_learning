from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="api key"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template="write a joke anout {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template = "explain the following joke: {joke}",
    input_variables=["joke"],
)

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

result = chain.invoke({"topic": "AI"}) 

print(result)


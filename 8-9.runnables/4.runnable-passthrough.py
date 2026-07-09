from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="write a joke anout {topic}",
    input_variables=["topic"],
)

prompt2 = PromptTemplate(
    template = "explain the following joke: {joke}",
    input_variables=["joke"],
)



joke_Chain = RunnableSequence(prompt1, model, parser)

parallel_Chain = RunnableParallel({
    'joke': RunnablePassthrough(joke_Chain),
    'explain': RunnableSequence(prompt2, model, parser)
})

final_Chain = RunnableSequence(joke_Chain, parallel_Chain)

result = final_Chain.invoke({"topic": "AI"})

print(result)
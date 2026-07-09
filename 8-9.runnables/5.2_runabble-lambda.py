from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough

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


def count_words(text):
    return len(text.split())

joke_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(joke_chain),
    'word_count': RunnableLambda(count_words)
})

fianl_chain = RunnableSequence(joke_chain, parallel_chain)

result = fianl_chain.invoke({"topic": "AI"})

fianal_result = """{} \n word count: {}""".format(result['joke'], result['word_count'])

print(fianal_result)


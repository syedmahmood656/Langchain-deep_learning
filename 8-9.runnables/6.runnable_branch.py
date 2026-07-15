from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence, RunnableParallel, RunnableLambda, RunnablePassthrough, RunnableBranch

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="HUGGINGFACEHUB_ACCESS_TOKEN"
)

model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt1 = PromptTemplate(
    template= 'generate a detailed report about {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template= 'generate a summary about {text}',
    input_variables=['text']
)

deatail_text_chain = RunnableSequence(prompt1, model, parser)

branch_runnbale_chain = RunnableBranch(
    (lambda x: len(x.split()) > 300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(deatail_text_chain, branch_runnbale_chain)

result = final_chain.invoke({'topic': 'AI'})

print(result)


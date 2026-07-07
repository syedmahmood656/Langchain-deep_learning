from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
)
model3 = ChatHuggingFace(llm=llm)

prompt1 = PromptTemplate(
    template="generate a short and concise summary of the following text: {text}", 
input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="generate a list of 5 questions based on the following text: {text}", 
input_variables=["text"]
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    prompt1 | model1 | parser, 
    prompt2 | model2 | parser
)

merged_chain = prompt3 | model3 | parser    

chain = parallel_chain | merged_chain

text = "" # You can provide any text here to generate a summary and a list of questions based on that text. The output will be merged into a single document.

result = chain.invoke({'text':text})

print(result)

chain.get_graph().print_ascii() # visulize the chain graph, you can install graphviz or grandalf to visualize the graph
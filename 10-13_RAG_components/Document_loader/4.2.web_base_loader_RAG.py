from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

parser = StrOutputParser()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    huggingfacehub_api_token=os.environ.get('HUGGINGFACEHUB_API_TOKEN')
)

model = ChatHuggingFace(llm=llm)

url = 'https://huggingface.co/spaces/ianpan/chest-x-ray-ai'

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))

content = docs[0].page_content

print(docs[0].page_content)

prompt1 = PromptTemplate(
    template='write the summery of the given text \n {text} \n\n summery : ',
    input_variables=['text']
)

chain = prompt1 | model | parser

result = chain.invoke({'text':content})

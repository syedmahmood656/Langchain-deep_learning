from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StringOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation", 
    )

model = ChatHuggingFace(llm=llm)

# 1st prompt -> detail report

template1 = PromptTemplate(
    template="write a detailed report on the following topic: {topic}",
    input_variables=["topic"]
    )

#2nd prompt -> summary

template2 = PromptTemplate(
    template="write a summary of the following report: {report}",
    input_variables=["report"]
    )

# here you can see how we can use the StringOutputParser to parse the output of the model and pass it to the next prompt in the chain
# this where output parser "string" comes in handy, it will parse the output of the model and pass it to the next prompt in the chain

parser = StringOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "PHYCOLOGYcal effects of social media on infants and toddlers"})

print(result)
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.invoke({"topic": "PHYCOLOGYcal effects of social media on infants and toddlers"})

result = model.invoke(prompt1)

prompt2 = template2.invoke({"report": result.content})

result = model.invoke(prompt2)

print(result.content)

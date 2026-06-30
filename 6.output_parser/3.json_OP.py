from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JSONOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation", 
    )

model = ChatHuggingFace(llm=llm)

parser = JSONOutputParser()

template = PromptTemplate(
    template = "give me name, city ,and age of a frictional caharacter \n {format_instructions} \n {input}",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)
 ###--------------------------------------------------------------
prompt = template.invoke({"input": "luffy from one piece"})

print(prompt)
# this will show what the prompt looks like, and how the format instructions are included in the prompt

result = model.invoke(prompt)

print(result.content) # this will show the output of the model, which should be a JSON string

fianl_result = parser.parse(result.content) # this will parse the output of the model and return a dictionary

print(fianl_result) # this will show the final result, which should be a dictionary with the name, city, and age of the frictional character
print(type(fianl_result)) # this will show the type of the final result, which should be a dictionary 
  ###------------------------------------------------------------

  ### ------------------------------- OR ------------------------------###

"""
  chain = template | model | parser

  result = chain.invoke({"input": "luffy from one piece"})

  print(result) # this will show the final result, which should be a dictionary with the name, city, and age of the frictional character
  """

## a flaw in json parser is that the parser cannot responde in a particular foramt\schema 
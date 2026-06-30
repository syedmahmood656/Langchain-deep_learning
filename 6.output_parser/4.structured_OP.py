from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv  
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation", 
    )

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name="fact_1", description="fact_1 about the topic"),
    ResponseSchema(name="fact_2", description="fact_2 about the topic"),
    ResponseSchema(name="fact_3", description="fact_3 about the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "give me 3 facts about the following topic \n {format_instructions} \n {input}",
    input_variables=["input"],  
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

prompt = template.invoke({"input": "luffy from one piece"})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result) # this will show the final result, which should be a dictionary with the 3 facts about the topic

# can also use the chain method to do the same thing

# drawback - no data validation, if the model does not respond in the correct format, the parser will throw an error


from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens = 100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("what is the capital of india")

print(result.content)

# while running this code an llm model will be downloded loclly and it will work on your ram and cpu/gpu 
# the downloded madel = model_id
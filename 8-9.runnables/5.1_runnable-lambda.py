from langchain_core.runnables import RunnableLambda

def word_counter(word):
    return len(word.split())

word_counter_runnable = RunnableLambda(word_counter)    

word_count = word_counter_runnable.invoke("This is a test sentence for counting words.")

print(word_count)

# here we have converted a simple function into a runnable using RunnableLambda. This allows us to use the function in a more structured way within the LangChain framework, enabling us to easily integrate it with other runnables and workflows
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="api key"
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    system_message := SystemMessage(content="You are a helpful assistant.")
]

while True:
    user_input = input("You: ")
    chat_history.append(("user", HumanMessage(content=user_input)))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(("assistant", AIMessage(content=result.content)))
    print(f"Chatbot: {result.content}")

    # problrm solved - we are now utilizing the built-in message features provided by the langchain library, which simplifies the management of chat history and improves the structure of our messages. By using the built-in message features, we can create more structured and meaningful interactions between the user and the chatbot, while also reducing the likelihood of errors in managing the chat history. This approach allows us to maintain a clear distinction between different types of messages (system, human, assistant) and ensures that our chat history is organized in a way that is easy to manage and understand. The use of SystemMessage, HumanMessage, and AIMessage classes from langchain_core.messages helps to structure our conversation and provides a clear framework for managing the flow of the conversation between the user and the chatbot.
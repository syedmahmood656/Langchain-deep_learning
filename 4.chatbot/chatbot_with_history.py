from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="api key"
)

model = ChatHuggingFace(llm=llm)

chat_history = []

while True:
    user_input = input("You: ")
    chat_history.append(("user", user_input))
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(chat_history)
    chat_history.append(("assistant", result.content))
    print(f"Chatbot: {result.content}")

    # To stop the chatbot, type "exit" or "quit".
    # here we are maintaining a chat history that includes both user inputs and chatbot responses. This allows the chatbot to have context for the conversation, enabling more coherent and relevant responses. The chat history is passed to the model during invocation, allowing it to generate responses based on the entire conversation history rather than just the latest user input.
    # This approach enhances the chatbot's ability to understand and respond to user queries in a more meaningful way, as it can reference previous interactions and maintain the flow of the conversation. The chat history is stored as a list of tuples, where each tuple contains the speaker ("user" or "assistant") and their respective message. This structure allows for easy management and retrieval of the conversation context when needed.
    # This simple chatbot will continue to interact with the user until they choose to exit. To run this code, make sure you have the necessary libraries installed and your Hugging Face API token set up in your environment variables. This chatbot uses the Meta Llama 3 8B Instruct model, which is designed for conversational tasks. You can replace it with any other compatible model from Hugging Face if you wish to explore different capabilities. This chatbot runs in a loop, allowing for continuous interaction. The user can type their messages, and the chatbot will respond accordingly. The loop will terminate when the user types "exit" or "quit". This chatbot runs in a terminal or command-line interface, making it easy to test and interact with. You can further enhance the chatbot by adding features such as context retention, multi-turn conversations, or integration with other APIs for additional functionality.

    ### problem with this approach is that we are manually managing the chat history as a list of tuples, which can become cumbersome and error-prone as the conversation grows. Additionally, we are not utilizing the built-in message features provided by the langchain library, which can simplify the management of chat history and improve the structure of our messages. By using the built-in message features, we can create more structured and meaningful interactions between the user and the chatbot, while also reducing the likelihood of errors in managing the chat history. solution - "chatbot_with_history_using_builtin_message-feature.py" file.
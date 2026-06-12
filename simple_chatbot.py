from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    task="text-generation",
    huggingfacehub_api_token="api key"
)

model = ChatHuggingFace(llm=llm)

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    result = model.invoke(user_input)
    print(f"Chatbot: {result.content}")

    # To stop the chatbot, type "exit" or "quit".
    # You can also implement additional commands or features as needed.'
    # This simple chatbot will continue to interact with the user until they choose to exit.
    # To run this code, make sure you have the necessary libraries installed and your Hugging Face API token set up in your environment variables.
    # this chatbot uses the Meta Llama 3 8B Instruct model, which is designed for conversational tasks. You can replace it with any other compatible model from Hugging Face if you wish to explore different capabilities.
    # this catbot runs in a loop, allowing for continuous interaction. The user can type their messages, and the chatbot will respond accordingly. The loop will terminate when the user types "exit" or "quit".
    # this charbot runs in a terminal or command-line interface, making it easy to test and interact with. You can further enhance the chatbot by adding features such as context retention, multi-turn conversations, or integration with other APIs for additional functionality.
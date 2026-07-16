import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

chat_history = []  # this list holds the whole conversation for this session

print("TutorGpt is ready. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    chat_history.append(("human", user_input))

    response = llm.invoke(chat_history)
    print("Tutor:", response.content, "\n")

    chat_history.append(("ai", response.content))

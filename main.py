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

def explain_zeroshot(topic):
    prompt = f"Explain the concept of '{topic}' in simple terms, as if teaching a student who is new to it. Keep it clear and beginner-friendly."
    response = llm.invoke(prompt)
    return response.content


def summarize_zeroshot(text):
    prompt = f"Summarize the following text in a few clear sentences, keeping only the key points:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

print("EduTutor is ready. Commands: /explain <topic>, /summarize <text>, or just chat. Type 'exit' to quit.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    elif user_input.startswith("/explain "):
        topic = user_input.replace("/explain ", "")
        result = explain_zeroshot(topic)
        print("Tutor:", result, "\n")

    elif user_input.startswith("/summarize "):
        text = user_input.replace("/summarize ", "")
        result = summarize_zeroshot(text)
        print("Tutor:", result, "\n")

    else:
        chat_history.append(("human", user_input))
        response = llm.invoke(chat_history)
        print("Tutor:", response.content, "\n")
        chat_history.append(("ai", response.content))
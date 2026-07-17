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

def generate_quiz_fewshot(topic, num_questions=3):
    prompt = f"""Generate a quiz on a given topic. Follow the exact format shown in these examples.

Example 1:
Topic: Gravity
Q1. What is gravity?
A1. A force that pulls objects toward each other, especially toward the center of the Earth.
Q2. What did Newton discover about gravity?
A2. That the force of gravity between two objects depends on their mass and the distance between them.

Example 2:
Topic: Photosynthesis
Q1. What is photosynthesis?
A1. The process by which plants use sunlight to convert carbon dioxide and water into food.
Q2. What gas is released during photosynthesis?
A2. Oxygen.

Now generate a quiz following the exact same format:
Topic: {topic}
Number of questions: {num_questions}
"""
    response = llm.invoke(prompt)
    return response.content

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
    
    elif user_input.startswith("/quiz "):
        topic = user_input.replace("/quiz ", "")
        result = generate_quiz_fewshot(topic)
        print("Tutor:\n", result, "\n")

    else:
        chat_history.append(("human", user_input))
        response = llm.invoke(chat_history)
        print("Tutor:", response.content, "\n")
        chat_history.append(("ai", response.content))


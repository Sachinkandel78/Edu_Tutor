import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)
explanation_template = PromptTemplate(
    input_variables=["topic", "level"],
    template="Explain the concept of '{topic}' in simple terms, suitable for a {level} student. Keep it clear and beginner-friendly."
)

quiz_template = PromptTemplate(
    input_variables=["topic", "num_questions"],
    template="""Generate a quiz on a given topic. Follow the exact format shown in these examples.

Example 1:
Topic: Gravity
Q1. What is gravity?
A1. A force that pulls objects toward each other, especially toward the center of the Earth.

Now generate a quiz following the exact same format:
Topic: {topic}
Number of questions: {num_questions}
"""
)

notes_template = PromptTemplate(
    input_variables=["topic"],
    template="Create concise revision notes on '{topic}' in bullet points, highlighting key terms in bold."
)

parser = StrOutputParser()

explanation_chain = explanation_template | llm | parser
notes_chain = notes_template | llm | parser
quiz_chain = quiz_template | llm | parser

study_plan_template = PromptTemplate(
    input_variables=["subject", "days", "hours_per_day"],
    template="Create a day-by-day study plan for '{subject}' spread over {days} days, assuming {hours_per_day} hours of study per day. Be specific about what to cover each day."
)


chat_history = []  # this list holds the whole conversation for this session

roles = {
    "teacher": "You are a patient and encouraging Teacher. Explain concepts simply, using relatable analogies, and check the student's understanding before moving on.",
    "examiner": "You are a formal Examiner. Ask questions, evaluate the student's answers critically, and give concise feedback with a score out of 10. Do not give away answers easily.",
    "coach": "You are an encouraging Study Coach. Focus on motivation, time management, and practical study strategies rather than deep subject explanations.",
    "expert": "You are a precise Subject Expert. Give detailed, technically accurate answers, using correct terminology, suitable for an advanced student."
}

current_role = "teacher"  # default role when the app starts

def explain_zeroshot(topic):
    prompt = f"Explain the concept of '{topic}' in simple terms, as if teaching a student who is new to it. Keep it clear and beginner-friendly."
    response = llm.invoke(prompt)
    return response.content


def summarize_zeroshot(text):
    prompt = f"Summarize the following text in a few clear sentences, keeping only the key points:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

print("EduTutor is ready. Commands: /explain <topic>, /summarize <text>, or just chat. Type 'exit' to quit.\n")
def explain_with_template(topic, level="beginner"):
    prompt = explanation_template.format(topic=topic, level=level)
    response = llm.invoke(prompt)
    return response.content


def generate_notes(topic):
    prompt = notes_template.format(topic=topic)
    response = llm.invoke(prompt)
    return response.content


def generate_study_plan(subject, days=3, hours_per_day=2):
    prompt = study_plan_template.format(subject=subject, days=days, hours_per_day=hours_per_day)
    response = llm.invoke(prompt)
    return response.content


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


def solve_stepwise(problem):
    prompt = f"""Solve the following problem. Think step by step, showing your reasoning clearly before giving the final answer.

Problem: {problem}

Reasoning:"""
    response = llm.invoke(prompt)
    return response.content

def full_learning_chain(topic):
    explanation = explanation_chain.invoke({"topic": topic, "level": "beginner"})
    notes = notes_chain.invoke({"topic": topic})
    quiz = quiz_chain.invoke({"topic": topic, "num_questions": 3})

    return f"""--- Explanation ---
{explanation}

--- Revision Notes ---
{notes}

--- Quiz ---
{quiz}
"""

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
    
    elif user_input.startswith("/solve "):
        problem = user_input.replace("/solve ", "")
        result = solve_stepwise(problem)
        print("Tutor:\n", result, "\n")

    elif user_input.startswith("/role "):
        new_role = user_input.replace("/role ", "").strip().lower()
        if new_role in roles:
            current_role = new_role
            print(f"Role switched to: {current_role}\n")
        else:
            print(f"Unknown role. Available roles: {', '.join(roles.keys())}\n")
    elif user_input.startswith("/notes "):
        topic = user_input.replace("/notes ", "")
        result = generate_notes(topic)
        print("Tutor:\n", result, "\n")

    elif user_input.startswith("/plan "):
        subject = user_input.replace("/plan ", "")
        result = generate_study_plan(subject)
        print("Tutor:\n", result, "\n")
    
    elif user_input.startswith("/learn "):
        topic = user_input.replace("/learn ", "")
        result = full_learning_chain(topic)
        print("Tutor:\n", result, "\n")

    else:
        messages = [("system", roles[current_role])] + chat_history + [("human", user_input)]
        response = llm.invoke(messages)
        print(f"Tutor ({current_role}):", response.content, "\n")
        chat_history.append(("human", user_input))
        chat_history.append(("ai", response.content))

 

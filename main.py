import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_agent
import json
from datetime import date

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile",
    temperature=0.7
)

PROFILE_FILE = "user_profile.json"

# ---------- Templates ----------

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

study_plan_template = PromptTemplate(
    input_variables=["subject", "days", "hours_per_day"],
    template="Create a day-by-day study plan for '{subject}' spread over {days} days, assuming {hours_per_day} hours of study per day. Be specific about what to cover each day."
)

# ---------- Chains (Step 7) ----------

parser = StrOutputParser()

explanation_chain = explanation_template | llm | parser
notes_chain = notes_template | llm | parser
quiz_chain = quiz_template | llm | parser


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

# ---------- Zero-shot functions (Step 2) ----------

def explain_zeroshot(topic):
    prompt = f"Explain the concept of '{topic}' in simple terms, as if teaching a student who is new to it. Keep it clear and beginner-friendly."
    response = llm.invoke(prompt)
    return response.content


def summarize_zeroshot(text):
    prompt = f"Summarize the following text in a few clear sentences, keeping only the key points:\n\n{text}"
    response = llm.invoke(prompt)
    return response.content

# ---------- Template-based functions (Step 6) ----------

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

# ---------- Few-shot function (Step 3) ----------

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

# ---------- Chain-of-Thought function (Step 4) ----------

def solve_stepwise(problem):
    prompt = f"""Solve the following problem. Think step by step, showing your reasoning clearly before giving the final answer.

Problem: {problem}

Reasoning:"""
    response = llm.invoke(prompt)
    return response.content

# ---------- Tools (Step 9) ----------
# NOTE: these are defined AFTER study_plan_template, generate_study_plan and
# summarize_zeroshot exist, since they depend on them.

@tool
def calculator(expression: str) -> str:
    """Useful for solving math expressions like '45 * 12' or '(200 - 50) / 5'. Input must be a valid math expression."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


@tool
def study_planner(subject_and_days: str) -> str:
    """Useful for creating a study plan. Input format: 'subject, number of days' e.g. 'Physics, 5'."""
    parts = subject_and_days.split(",")
    subject = parts[0].strip()
    days = parts[1].strip() if len(parts) > 1 else "3"
    return generate_study_plan(subject, days=days, hours_per_day=2)


@tool
def summarizer_tool(text: str) -> str:
    """Useful for summarizing a block of text into a few key sentences."""
    return summarize_zeroshot(text)


tools = [calculator, study_planner, summarizer_tool]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful academic tutor assistant. "
        "Use tools when the user's request needs calculation, study planning, or summarization. "
        "When a tool returns a result, include the FULL tool output in your final response to the user — "
        "do not just acknowledge it, actually show the complete content. "
        "Otherwise, answer directly without using a tool."
    )
)

# ---------- Roles (Step 5) ----------

roles = {
    "teacher": "You are a patient and encouraging Teacher. Explain concepts simply, using relatable analogies, and check the student's understanding before moving on.",
    "examiner": "You are a formal Examiner. Ask questions, evaluate the student's answers critically, and give concise feedback with a score out of 10. Do not give away answers easily.",
    "coach": "You are an encouraging Study Coach. Focus on motivation, time management, and practical study strategies rather than deep subject explanations.",
    "expert": "You are a precise Subject Expert. Give detailed, technically accurate answers, using correct terminology, suitable for an advanced student."
}

# ---------- Memory (Step 8) ----------

def load_profile():
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            return json.load(f)
    else:
        return {
            "name": None,
            "topics_studied": [],
            "preferred_role": "teacher",
            "quiz_history": []
        }


def save_profile(profile):
    with open(PROFILE_FILE, "w") as f:
        json.dump(profile, f, indent=2)


chat_history = []  # session memory

profile = load_profile()
current_role = profile.get("preferred_role", "teacher")

print("EduTutor is ready. Commands: /explain, /summarize, /quiz, /solve, /role, /notes, /plan, /learn, /ask, or just chat. Type 'exit' to quit.\n")

if profile["name"]:
    print(f"Welcome back, {profile['name']}!")
    if profile["topics_studied"]:
        last_topic = profile["topics_studied"][-1]["topic"]
        print(f"Last time, we studied: {last_topic}\n")
else:
    profile["name"] = input("What's your name? ")
    save_profile(profile)

# ---------- Main loop (Step 10 wiring) ----------

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    elif user_input.startswith("/explain "):
        topic = user_input.replace("/explain ", "")
        result = explain_zeroshot(topic)
        print("Tutor:", result, "\n")

        profile["topics_studied"].append({"topic": topic, "date": str(date.today())})
        save_profile(profile)

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
            profile["preferred_role"] = current_role
            save_profile(profile)
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

        profile["topics_studied"].append({"topic": topic, "date": str(date.today())})
        save_profile(profile)

    elif user_input.startswith("/ask "):
        query = user_input.replace("/ask ", "")
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        final_message = result["messages"][-1].content
        print("Tutor:", final_message, "\n")

    else:
        messages = [("system", roles[current_role])] + chat_history + [("human", user_input)]
        response = llm.invoke(messages)
        print(f"Tutor ({current_role}):", response.content, "\n")
        chat_history.append(("human", user_input))
        chat_history.append(("ai", response.content))
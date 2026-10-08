
import os
import json
import sqlite3
import uuid
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="StudyLens AI",
    page_icon="🎓",
    layout="wide"
)

DATABASE = "studylens.db"
AI_MODEL = "gpt-4.1-mini"

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    max-width: 1150px;
}
h1, h2, h3 {
    letter-spacing: -0.5px;
}
.stButton > button[kind="primary"] {
    border-radius: 10px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

def connect_db():
    return sqlite3.connect(
        DATABASE,
        timeout=15
    )


def init_db():
    with connect_db() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                student TEXT,
                subject TEXT,
                topic TEXT,
                question TEXT,
                selected TEXT,
                correct TEXT,
                is_correct INTEGER,
                confidence TEXT,
                created_at TEXT
            )
        """)


def save_results(student, subject, results):
    session_id = str(uuid.uuid4())
    timestamp = datetime.now().isoformat(
        timespec="seconds"
    )

    rows = []

    for result in results:
        rows.append((
            session_id,
            student,
            subject,
            result["topic"],
            result["question"],
            result["selected"],
            result["correct"],
            int(result["is_correct"]),
            result["confidence"],
            timestamp
        ))

    with connect_db() as con:
        con.executemany("""
            INSERT INTO attempts (
                session_id,
                student,
                subject,
                topic,
                question,
                selected,
                correct,
                is_correct,
                confidence,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)


def load_results(student):
    with connect_db() as con:
        return pd.read_sql_query(
            """
            SELECT * FROM attempts
            WHERE student = ?
            ORDER BY id
            """,
            con,
            params=(student,)
        )


init_db()

def get_api_key():
    key = os.getenv("OPENAI_API_KEY", "")

    if key:
        return key

    try:
        return st.secrets.get("OPENAI_API_KEY", "")
    except (FileNotFoundError, KeyError):
        return ""


def ai_request(instructions, content, json_mode=False):
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError("AI engine is not configured.")

    client = OpenAI(api_key=api_key)

    arguments = {
        "model": AI_MODEL,
        "instructions": instructions,
        "input": content,
        "max_output_tokens": 4000
    }

    if json_mode:
        arguments["text"] = {
            "format": {"type": "json_object"}
        }

    response = client.responses.create(**arguments)

    if not response.output_text:
        raise RuntimeError("Empty response.")

    if json_mode:
        return json.loads(response.output_text)

    return response.output_text

QUESTION_BANK = [
    {
        "topic": "Quadratic Equations",
        "question": "Find the roots of x² - 5x + 6 = 0.",
        "options": [
            "2 and 3",
            "1 and 6",
            "-2 and -3",
            "0 and 5"
        ],
        "answer": "2 and 3",
        "explanation": "(x - 2)(x - 3) = 0."
    },
    {
        "topic": "Quadratic Equations",
        "question": "Find the discriminant of x² + 4x + 4 = 0.",
        "options": ["0", "4", "8", "16"],
        "answer": "0",
        "explanation": "D = b² - 4ac = 16 - 16 = 0."
    },
    {
        "topic": "Quadratic Equations",
        "question": "A quadratic equation with a negative discriminant has:",
        "options": [
            "Two real roots",
            "One repeated real root",
            "No real roots",
            "Infinitely many roots"
        ],
        "answer": "No real roots",
        "explanation": "A negative discriminant gives non-real roots."
    },
    {
        "topic": "Trigonometry",
        "question": "What is sin 30°?",
        "options": ["0", "1/2", "1", "√3/2"],
        "answer": "1/2",
        "explanation": "The standard value of sin 30° is 1/2."
    },
    {
        "topic": "Trigonometry",
        "question": "Simplify sin²θ + cos²θ.",
        "options": ["0", "1", "2", "tan θ"],
        "answer": "1",
        "explanation": "This follows from the Pythagorean identity."
    },
    {
        "topic": "Trigonometry",
        "question": "What is tan 45°?",
        "options": ["0", "1", "√3", "Undefined"],
        "answer": "1",
        "explanation": "tan 45° = sin 45° / cos 45° = 1."
    },
    {
        "topic": "Sets",
        "question": "If A = {1, 2} and B = {2, 3}, find A ∩ B.",
        "options": [
            "{1, 3}",
            "{2}",
            "{1, 2, 3}",
            "{}"
        ],
        "answer": "{2}",
        "explanation": "The intersection contains common elements."
    },
    {
        "topic": "Sets",
        "question": "How many subsets does a set of 3 elements have?",
        "options": ["3", "6", "8", "9"],
        "answer": "8",
        "explanation": "Number of subsets = 2³ = 8."
    },
    {
        "topic": "Sets",
        "question": "What is A ∪ ∅?",
        "options": ["A", "∅", "Universal set", "Undefined"],
        "answer": "A",
        "explanation": "The union of A with the empty set is A."
    }
]


def validate_questions(data):
    questions = data.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError("Invalid question format.")

    valid = []

    for q in questions:
        if not isinstance(q, dict):
            continue

        required = [
            "topic", "question",
            "options", "answer", "explanation"
        ]

        if not all(k in q for k in required):
            continue

        if not all(
            isinstance(q[k], str) and q[k].strip()
            for k in ["topic", "question", "answer", "explanation"]
        ):
            continue

        options = q["options"]

        if not isinstance(options, list):
            continue

        if len(options) != 4:
            continue

        if not all(
            isinstance(option, str) and option.strip()
            for option in options
        ):
            continue

        if len(set(options)) != 4:
            continue

        if q["answer"] not in options:
            continue

        valid.append(q)

    if not valid:
        raise ValueError("No valid questions were generated.")

    return valid


def generate_questions(subject, chapter, focus="", count=8):
    instructions = """
You are the StudyLens educational question engine.

Generate accurate diagnostic multiple-choice questions
for a Class 11 student in the requested subject.

Return a valid JSON object with a "questions" array.

Every question must contain:
topic: specific concept being tested
question: clear question text
options: exactly four different answer strings
answer: exact text of the correct option
explanation: short, accurate reasoning

Requirements:
- Generate the requested number of questions.
- Use varied concepts and difficulty levels.
- Avoid ambiguous questions.
- Avoid duplicate questions.
- Check factual and mathematical correctness.
- Focus on weaknesses when provided.
- Do not use Markdown outside the JSON.
"""

    payload = {
        "grade": 11,
        "subject": subject,
        "chapter": chapter,
        "focus": focus,
        "question_count": count
    }

    response = ai_request(
        instructions,
        json.dumps(payload),
        json_mode=True
    )

    return validate_questions(response)

def analyse_results(results):
    df = pd.DataFrame(results)

    if df.empty:
        return pd.DataFrame()

    summary = df.groupby("topic").agg(
        attempted=("is_correct", "count"),
        correct=("is_correct", "sum")
    ).reset_index()

    summary["accuracy"] = (
        summary["correct"]
        / summary["attempted"]
        * 100
    ).round(1)

    summary["priority"] = summary["accuracy"].apply(
        lambda score:
        "High" if score < 50
        else "Medium" if score < 80
        else "Low"
    )

    return summary.sort_values("accuracy")


def create_learning_report(subject, results):
    summary = analyse_results(results)

    payload = {
        "subject": subject,
        "concept_scores": summary.to_dict("records"),
        "student_answers": results
    }

    instructions = """
You are StudyLens AI, a personalised learning coach.

Analyse the supplied assessment evidence.

Write a clear student-friendly report containing:

1. Overall learning assessment
2. Strengths demonstrated
3. Potential conceptual weaknesses
4. Why incorrect answers are incorrect
5. Personalised three-day revision timetable
6. Recommended next steps

Important:
- Do not invent scores or performance history.
- Treat weaknesses as preliminary indications.
- Clearly explain mathematical or factual mistakes.
- Keep explanations suitable for Class 11.
- Use Markdown formatting.
"""

    return ai_request(
        instructions,
        json.dumps(payload, default=str)
    )


# ==========================================
# SESSION MANAGEMENT
# ==========================================

defaults = {
    "quiz": [],
    "quiz_id": "",
    "quiz_subject": "",
    "results": None,
    "report": "",
    "tutor_reply": "",
    "student": "Demo Student"
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def prepare_quiz(questions, subject):
    st.session_state.quiz = questions
    st.session_state.quiz_subject = subject
    st.session_state.quiz_id = str(uuid.uuid4())
    st.session_state.results = None
    st.session_state.report = ""
    st.session_state.tutor_reply = ""


# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🎓 StudyLens AI")

student = st.sidebar.text_input(
    "Student / Demo ID",
    value=st.session_state.student
).strip() or "Demo Student"

st.session_state.student = student

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Diagnostic Test",
        "Learning Dashboard",
        "AI Learning Coach",
        "Exhibition Demo"
    ]
)

st.sidebar.divider()

if get_api_key():
    st.sidebar.success("StudyLens Engine: Online")
else:
    st.sidebar.warning("StudyLens Engine: Demo Mode")


# ==========================================
# HOME
# ==========================================

if page == "Home":
    st.title("🎓 StudyLens AI")

    st.subheader(
        "Discover what you don't understand."
    )

    st.write("""
    **StudyLens AI** is a personalised learning diagnostic
    platform that helps students identify weak concepts,
    understand their mistakes and improve strategically.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Supported Subjects", "Any")
    col2.metric("Starter Questions", len(QUESTION_BANK))
    col3.metric("Personalised Guidance", "AI")

    st.divider()

    st.markdown("### How StudyLens Works")

    st.markdown("""
    **1. Take a diagnostic test**

    Choose a subject and answer targeted questions.

    **2. Discover learning gaps**

    StudyLens identifies concepts that may require revision.

    **3. Understand your mistakes**

    Receive explanations and personalised learning insights.

    **4. Follow a revision plan**

    Work on weak concepts and attempt practice questions.

    **5. Track improvement**

    Monitor your performance over time.
    """)



elif page == "Diagnostic Test":
    st.title("📝 Diagnostic Assessment")

    mode = st.radio(
        "Select assessment type",
        [
            "Verified Mathematics Demo",
            "Create Custom AI Assessment"
        ]
    )

    if mode == "Verified Mathematics Demo":
        subject = "Mathematics"
        chapter = "Mixed Topics"

        st.info(
            "This demonstration uses nine prechecked "
            "Class 11 Mathematics questions."
        )

    else:
        subject = st.text_input(
            "Subject",
            placeholder="Mathematics, Biology, History, Economics..."
        ).strip()

        chapter = st.text_input(
            "Chapter or topic",
            placeholder="Enter any chapter or concept"
        ).strip()

    if st.button("Generate Assessment", type="primary"):
        if mode == "Verified Mathematics Demo":
            prepare_quiz(QUESTION_BANK.copy(), subject)
            st.success("Assessment ready.")

        elif not subject or not chapter:
            st.warning("Please enter the subject and chapter.")

        else:
            try:
                with st.spinner(
                    "StudyLens is preparing your assessment..."
                ):
                    questions = generate_questions(
                        subject,
                        chapter
                    )

                prepare_quiz(questions, subject)
                st.success("Assessment generated successfully.")

            except Exception:
                st.error(
                    "Assessment generation is unavailable. "
                    "Please check the AI configuration and try again."
                )

    questions = st.session_state.quiz

    if questions and st.session_state.results is None:
        st.subheader(
            f"Assessment: {st.session_state.quiz_subject}"
        )

        with st.form(
            f"assessment_{st.session_state.quiz_id}"
        ):
            selections = []
            confidences = []

            for i, q in enumerate(questions):
                st.markdown(f"### Question {i + 1}")
                st.write(q["question"])

                answer = st.radio(
                    "Choose one answer",
                    q["options"],
                    index=None,
                    key=f"{st.session_state.quiz_id}_q{i}"
                )

                confidence = st.select_slider(
                    "How confident are you?",
                    ["Unsure", "Somewhat Sure", "Confident"],
                    value="Somewhat Sure",
                    key=f"{st.session_state.quiz_id}_c{i}"
                )

                selections.append(answer)
                confidences.append(confidence)

                st.divider()

            submitted = st.form_submit_button(
                "Submit Assessment",
                type="primary"
            )

        if submitted:
            if any(a is None for a in selections):
                st.warning("Please answer every question.")

            else:
                results = []

                for q, selected, confidence in zip(
                    questions,
                    selections,
                    confidences
                ):
                    results.append({
                        "topic": q["topic"],
                        "question": q["question"],
                        "selected": selected,
                        "correct": q["answer"],
                        "is_correct": selected == q["answer"],
                        "confidence": confidence,
                        "explanation": q["explanation"]
                    })

                save_results(
                    student,
                    st.session_state.quiz_subject,
                    results
                )

                st.session_state.results = results
                st.rerun()

    if st.session_state.results is not None:
        results = st.session_state.results

        correct = sum(
            r["is_correct"] for r in results
        )

        total = len(results)
        percentage = round(correct / total * 100, 1)

        st.subheader("Your Assessment Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Score", f"{percentage}%")
        c2.metric("Correct Answers", correct)
        c3.metric("Total Questions", total)

        st.progress(percentage / 100)

        summary = analyse_results(results)

        st.subheader("Concept-wise Performance")

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        fig = px.bar(
            summary,
            x="topic",
            y="accuracy",
            color="accuracy",
            range_y=[0, 100],
            color_continuous_scale="RdYlGn",
            labels={
                "topic": "Concept",
                "accuracy": "Accuracy (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Answer Review")

        for i, r in enumerate(results):
            icon = "✅" if r["is_correct"] else "❌"

            with st.expander(
                f"{icon} Question {i + 1}: {r['topic']}"
            ):
                st.write(r["question"])
                st.write("**Your answer:**", r["selected"])
                st.write("**Correct answer:**", r["correct"])
                st.write("**Explanation:**", r["explanation"])

        st.info(
            "This is a preliminary diagnostic assessment. "
            "More questions may be needed to confirm a learning gap."
        )

        if st.button("Generate Personalised AI Report"):
            try:
                with st.spinner(
                    "StudyLens is analysing your performance..."
                ):
                    st.session_state.report = create_learning_report(
                        st.session_state.quiz_subject,
                        results
                    )

            except Exception:
                st.error(
                    "The learning report could not be generated. "
                    "Check your AI connection and try again."
                )

        if st.session_state.report:
            st.markdown(st.session_state.report)


elif page == "Learning Dashboard":
    st.title("📊 Learning Analytics")

    history = load_results(student)

    if history.empty:
        st.info(
            "Complete a diagnostic assessment to "
            "unlock your learning dashboard."
        )

    else:
        total = len(history)
        correct = int(history["is_correct"].sum())
        accuracy = correct / total * 100

        c1, c2, c3 = st.columns(3)

        c1.metric("Questions Attempted", total)
        c2.metric("Correct Answers", correct)
        c3.metric("Overall Accuracy", f"{accuracy:.1f}%")

        st.divider()

        summary = history.groupby(
            ["subject", "topic"],
            as_index=False
        ).agg(
            attempted=("is_correct", "count"),
            correct=("is_correct", "sum")
        )

        summary["accuracy"] = (
            summary["correct"]
            / summary["attempted"]
            * 100
        )

        summary["concept"] = (
            summary["subject"] + " | " + summary["topic"]
        )

        st.subheader("Concept Performance")

        fig = px.bar(
            summary,
            x="concept",
            y="accuracy",
            color="accuracy",
            color_continuous_scale="RdYlGn",
            range_y=[0, 100],
            labels={
                "concept": "Concept",
                "accuracy": "Accuracy (%)"
            }
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Revision Priorities")

        weak_topics = summary.sort_values(
            "accuracy"
        ).head(5)

        st.dataframe(
            weak_topics[
                ["subject", "topic", "attempted", "accuracy"]
            ],
            hide_index=True,
            use_container_width=True
        )

        st.subheader("Assessment History")

        assessment_history = history.groupby(
            ["session_id", "subject", "created_at"],
            as_index=False
        ).agg(
            questions=("is_correct", "count"),
            correct=("is_correct", "sum")
        )

        assessment_history["accuracy"] = (
            assessment_history["correct"]
            / assessment_history["questions"]
            * 100
        )

        assessment_history = assessment_history.sort_values(
            "created_at"
        )

        assessment_history["attempt_number"] = range(
            1, len(assessment_history) + 1
        )

        trend = px.line(
            assessment_history,
            x="attempt_number",
            y="accuracy",
            markers=True,
            range_y=[0, 100],
            hover_data=["subject", "created_at"],
            labels={
                "attempt_number": "Assessment Number",
                "accuracy": "Accuracy (%)"
            }
        )

        st.plotly_chart(
            trend,
            use_container_width=True
        )

        st.caption(
            "Assessments may contain different questions "
            "and difficulty levels. Trends are indicative."
        )

        st.download_button(
            "Download Learning Data",
            data=history.to_csv(index=False),
            file_name="studylens_learning_data.csv",
            mime="text/csv"
        )



elif page == "AI Learning Coach":
    st.title("🤖 StudyLens Learning Coach")

    history = load_results(student)

    if history.empty:
        st.info(
            "Take an assessment first to activate "
            "personalised learning support."
        )

    else:
        performance = history.groupby(
            ["subject", "topic"],
            as_index=False
        ).agg(
            attempted=("is_correct", "count"),
            correct=("is_correct", "sum")
        )

        performance["accuracy"] = (
            performance["correct"]
            / performance["attempted"]
            * 100
        )

        performance = performance.sort_values("accuracy")

        options = [
            (row["subject"], row["topic"])
            for _, row in performance.iterrows()
        ]

        chosen = st.selectbox(
            "Select a concept",
            options,
            format_func=lambda item: f"{item[0]} — {item[1]}"
        )

        subject, topic = chosen

        action = st.radio(
            "What do you need help with?",
            [
                "Explain the concept",
                "Create a 3-day revision plan",
                "Generate targeted practice questions"
            ]
        )

        if st.button("Get Personalised Assistance", type="primary"):
            try:
                with st.spinner(
                    "StudyLens is preparing your guidance..."
                ):
                    if action == "Generate targeted practice questions":
                        questions = generate_questions(
                            subject,
                            topic,
                            focus=topic,
                            count=5
                        )

                        prepare_quiz(questions, subject)

                        st.session_state.tutor_reply = (
                            "Your targeted practice assessment is ready. "
                            "Go to **Diagnostic Test** to attempt it."
                        )

                    else:
                        relevant = history[
                            (history["subject"] == subject)
                            & (history["topic"] == topic)
                        ]

                        answer_data = relevant[
                            [
                                "question",
                                "selected",
                                "correct",
                                "is_correct",
                                "confidence"
                            ]
                        ].tail(12).to_dict("records")

                        if action == "Explain the concept":
                            instruction = """
You are StudyLens AI, a patient educational coach.

Explain the selected Class 11 concept clearly.
Use examples, explain mistakes, and suggest
a few practice exercises.

Base personalised claims on actual student answers.
"""
                        else:
                            instruction = """
You are StudyLens AI, a learning planner.

Produce a personalised three-day revision plan
for this concept.

Include daily objectives, practice tasks,
estimated study time and short self-tests.

Do not invent student performance.
"""

                        st.session_state.tutor_reply = ai_request(
                            instruction,
                            json.dumps({
                                "subject": subject,
                                "concept": topic,
                                "recent_answers": answer_data
                            })
                        )

            except Exception:
                st.error(
                    "StudyLens could not complete the request. "
                    "Please check the connection and try again."
                )

        if st.session_state.tutor_reply:
            st.markdown(st.session_state.tutor_reply)



elif page == "Exhibition Demo":
    st.title("🎪 The StudyLens Challenge")

    st.subheader("Same Marks. Different Learning Gaps.")

    st.write("""
    Two students both score **60%** on the same
    Mathematics assessment.

    But should they follow the same revision plan?

    **StudyLens shows why the answer is no.**
    """)

    demo = pd.DataFrame({
        "Concept": [
            "Algebra",
            "Trigonometry",
            "Sets",
            "Functions",
            "Sequences"
        ],
        "Student A": [100, 100, 50, 50, 0],
        "Student B": [0, 50, 50, 100, 100]
    })

    left, right = st.columns(2)

    with left:
        st.metric("Student A", "60%")

        fig_a = px.bar(
            demo,
            x="Concept",
            y="Student A",
            range_y=[0, 100],
            color="Student A",
            color_continuous_scale="RdYlGn"
        )

        st.plotly_chart(
            fig_a,
            use_container_width=True
        )

        st.warning(
            "Priority: Sequences, Functions and Sets."
        )

    with right:
        st.metric("Student B", "60%")

        fig_b = px.bar(
            demo,
            x="Concept",
            y="Student B",
            range_y=[0, 100],
            color="Student B",
            color_continuous_scale="RdYlGn"
        )

        st.plotly_chart(
            fig_b,
            use_container_width=True
        )

        st.warning(
            "Priority: Algebra, Trigonometry and Sets."
        )

    st.divider()

    st.success("""
    Identical marks do not imply identical learning needs.

    StudyLens AI uses concept-level assessment
    to recommend more personalised revision.
    """)

    st.caption(
        "Illustrative data. Each concept is assumed "
        "to have equal assessment weight."
    )



st.divider()

st.caption(
    "© 2026 StudyLens AI | "
    "Personalised Learning Intelligence Platform | "
    "Exhibition Prototype"
)

st.caption(
    "AI-generated educational material may contain "
    "errors and should be independently verified. "
    "Learning-gap predictions are preliminary."
)

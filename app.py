
import os
import json
import sqlite3
import uuid
from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st
from google import genai
from google.genai import types


# ==========================================
# STUDYLENS AI - APPLICATION SETTINGS
# ==========================================

st.set_page_config(
    page_title="StudyLens AI",
    page_icon="🎓",
    layout="wide"
)

DATABASE = "studylens.db"

# Model can also be configured in Streamlit Secrets.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


# ==========================================
# INTERFACE STYLING
# ==========================================

st.markdown("""
<style>
.block-container {
    max-width: 1150px;
    padding-top: 2rem;
}

h1, h2, h3 {
    letter-spacing: -0.5px;
}

.stButton > button {
    border-radius: 10px;
    font-weight: 600;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 12px;
    padding: 14px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# DATABASE FUNCTIONS
# ==========================================

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


# ==========================================
# GEMINI INTELLIGENCE ENGINE
# ==========================================

def get_setting(name, default=""):
    value = os.getenv(name, "").strip()

    if value:
        return value

    try:
        return str(
            st.secrets.get(name, default)
        ).strip()
    except (FileNotFoundError, KeyError):
        return default


def get_api_key():
    return get_setting("GEMINI_API_KEY")


def get_model():
    return get_setting(
        "GEMINI_MODEL",
        DEFAULT_MODEL
    )


def ai_request(instructions, content, json_mode=False):
    """
    Central intelligence engine.

    Used by:
    - AI assessments
    - Learning reports
    - Study plans
    - Targeted practice
    - Connection diagnostics
    """

    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "AI key missing. Add GEMINI_API_KEY "
            "to Streamlit Secrets."
        )

    config = types.GenerateContentConfig(
        system_instruction=instructions,
        temperature=0.3,
        max_output_tokens=5000,
        response_mime_type=(
            "application/json"
            if json_mode
            else "text/plain"
        )
    )

    with genai.Client(api_key=api_key) as client:
        response = client.models.generate_content(
            model=get_model(),
            contents=content,
            config=config
        )

    response_text = response.text

    if not response_text:
        raise RuntimeError(
            "The intelligence engine returned "
            "an empty response."
        )

    if json_mode:
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "The AI returned invalid JSON."
            ) from exc

        if not isinstance(result, dict):
            raise ValueError(
                "The AI response must be a JSON object."
            )

        return result

    return response_text


def readable_ai_error(error):
    """
    Convert technical errors into safe,
    visitor-friendly messages.
    """

    message = str(error).lower()

    if "api key" in message or "api_key" in message:
        return (
            "The AI key is missing or invalid. "
            "Check your Streamlit Secrets."
        )

    if "429" in message or "resource_exhausted" in message:
        return (
            "The AI usage limit has been reached. "
            "Please try again later or check your quota."
        )

    if "404" in message or "not found" in message:
        return (
            "The selected AI model is unavailable "
            "for this project. Check GEMINI_MODEL."
        )

    if "403" in message or "permission_denied" in message:
        return (
            "Access was denied. Check your API key, "
            "project permissions and regional availability."
        )

    if "400" in message or "invalid_argument" in message:
        return (
            "The AI could not process the request. "
            "Check the model and request configuration."
        )

    return (
        "StudyLens could not complete the AI request. "
        "Please try again."
    )


# ==========================================
# BUILT-IN VERIFIED DEMO QUESTIONS
# ==========================================

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
        "question": "A quadratic with a negative discriminant has:",
        "options": [
            "Two distinct real roots",
            "One repeated real root",
            "No real roots",
            "Infinitely many roots"
        ],
        "answer": "No real roots",
        "explanation": "A negative discriminant means no real roots."
    },
    {
        "topic": "Trigonometry",
        "question": "What is sin 30°?",
        "options": ["0", "1/2", "1", "√3/2"],
        "answer": "1/2",
        "explanation": "sin 30° = 1/2."
    },
    {
        "topic": "Trigonometry",
        "question": "Simplify sin²θ + cos²θ.",
        "options": ["0", "1", "2", "tan θ"],
        "answer": "1",
        "explanation": "This is the Pythagorean identity."
    },
    {
        "topic": "Trigonometry",
        "question": "What is tan 45°?",
        "options": ["0", "1", "√3", "Undefined"],
        "answer": "1",
        "explanation": "tan 45° = 1."
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
        "explanation": "The common element is 2."
    },
    {
        "topic": "Sets",
        "question": "How many subsets does a set of 3 elements have?",
        "options": ["3", "6", "8", "9"],
        "answer": "8",
        "explanation": "A set with n elements has 2^n subsets."
    },
    {
        "topic": "Sets",
        "question": "What is A ∪ ∅?",
        "options": [
            "A",
            "∅",
            "Universal set",
            "Undefined"
        ],
        "answer": "A",
        "explanation": "Union with the empty set leaves A unchanged."
    }
]


# ==========================================
# AI QUESTION GENERATION
# ==========================================

def validate_questions(data):
    questions = data.get("questions", [])

    if not isinstance(questions, list):
        raise ValueError("Invalid question format.")

    cleaned = []

    for q in questions:
        if not isinstance(q, dict):
            continue

        required = [
            "topic", "question", "options",
            "answer", "explanation"
        ]

        if not all(key in q for key in required):
            continue

        if not all(
            isinstance(q[key], str) and q[key].strip()
            for key in [
                "topic", "question",
                "answer", "explanation"
            ]
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

        cleaned.append({
            "topic": q["topic"].strip(),
            "question": q["question"].strip(),
            "options": options,
            "answer": q["answer"],
            "explanation": q["explanation"].strip()
        })

    if not cleaned:
        raise ValueError(
            "No usable questions were generated."
        )

    return cleaned


def generate_questions(
    subject,
    chapter,
    focus="",
    count=8
):
    instructions = """
You are StudyLens AI, an educational assessment engine.

Create Class 11 diagnostic multiple-choice questions.

Return a valid JSON object with this structure:

{
  "questions": [
    {
      "topic": "Specific concept",
      "question": "Question text",
      "options": ["A", "B", "C", "D"],
      "answer": "A",
      "explanation": "Correct reasoning"
    }
  ]
}

Requirements:
- Generate the requested number of questions.
- Use exactly four distinct options per question.
- The answer must exactly match one option.
- Use appropriate Class 11 difficulty.
- Include several underlying concepts where suitable.
- Avoid duplicate or ambiguous questions.
- Check calculations and factual correctness.
- Keep explanations concise and informative.
- Focus on the specified weakness when supplied.
- Return JSON only.
"""

    payload = {
        "grade": 11,
        "subject": subject,
        "chapter": chapter,
        "focus": focus,
        "number_of_questions": count
    }

    response = ai_request(
        instructions,
        json.dumps(payload),
        json_mode=True
    )

    questions = validate_questions(response)

    return questions


# ==========================================
# SCORING AND LEARNING GAP ANALYSIS
# ==========================================

def analyse_results(results):
    df = pd.DataFrame(results)

    if df.empty:
        return pd.DataFrame()

    summary = df.groupby(
        "topic", as_index=False
    ).agg(
        attempted=("is_correct", "count"),
        correct=("is_correct", "sum")
    )

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

Analyse the student's diagnostic results.

Write a student-friendly report containing:

1. Overall assessment
2. Strengths demonstrated
3. Possible conceptual weaknesses
4. Explanations for incorrect answers
5. Personalised three-day revision plan
6. Recommended follow-up practice

Important:
- Do not invent scores or assessment history.
- Consider the student's confidence in answers.
- Treat suspected gaps as preliminary.
- Do not assume one mistake proves weak understanding.
- Use clear Markdown headings.
- Keep advice suitable for Class 11.
"""

    return ai_request(
        instructions,
        json.dumps(payload, default=str)
    )


# ==========================================
# SESSION STATE
# ==========================================

defaults = {
    "quiz": [],
    "quiz_id": "",
    "quiz_subject": "",
    "results": None,
    "report": "",
    "tutor_reply": "",
    "student": "Demo Student",
    "quiz_notice": ""
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
    st.sidebar.info("StudyLens AI: Key Configured")
else:
    st.sidebar.warning("StudyLens AI: Demo Mode")


# ==========================================
# AI CONNECTION DIAGNOSTICS
# ==========================================

with st.sidebar.expander("System Diagnostics"):
    st.caption("Check whether the online engine is working.")

    if st.button("Test AI Connection"):
        if not get_api_key():
            st.error(
                "No AI key found. Add GEMINI_API_KEY "
                "in Streamlit Secrets."
            )
        else:
            try:
                with st.spinner("Testing connection..."):
                    reply = ai_request(
                        "You are StudyLens AI.",
                        "Reply with exactly: StudyLens AI connected."
                    )

                st.success("AI connection successful.")
                st.write(reply)

            except Exception as error:
                st.error(readable_ai_error(error))

    if not get_api_key():
        st.caption(
            "The built-in quiz works without an AI connection."
        )


# ==========================================
# HOME PAGE
# ==========================================

if page == "Home":
    st.title("🎓 StudyLens AI")

    st.subheader(
        "Discover your gaps. Unlock your potential."
    )

    st.write("""
    StudyLens AI helps students identify potential
    conceptual weaknesses, understand mistakes
    and practise more strategically.

    **Traditional tests tell you how much you scored.
    StudyLens helps you decide what to study next.**
    """)

    c1, c2, c3 = st.columns(3)

    c1.metric("Supported Subjects", "Any")
    c2.metric("Starter Questions", len(QUESTION_BANK))
    c3.metric("Learning Guidance", "Personalised")

    st.divider()

    st.subheader("How It Works")

    st.markdown("""
    **1. Select a subject**

    Choose Mathematics, Physics, Chemistry,
    Biology, History or another subject.

    **2. Take a diagnostic assessment**

    Answer questions and record your confidence.

    **3. Discover possible learning gaps**

    StudyLens calculates performance by concept.

    **4. Receive personalised AI guidance**

    Understand mistakes and get a revision plan.

    **5. Practise and improve**

    Generate targeted questions and track progress.
    """)

    st.info(
        "For a quick demonstration, start with "
        "Diagnostic Test → Verified Mathematics Demo."
    )


# ==========================================
# DIAGNOSTIC TEST PAGE
# ==========================================

elif page == "Diagnostic Test":
    st.title("📝 Diagnostic Assessment")

    mode = st.radio(
        "Assessment Type",
        [
            "Verified Mathematics Demo",
            "Create Custom AI Assessment"
        ]
    )

    if mode == "Verified Mathematics Demo":
        subject = "Mathematics"
        chapter = "Mixed Class 11 Topics"

        st.info(
            "Nine verified questions covering "
            "Quadratics, Trigonometry and Sets."
        )

    else:
        subject = st.text_input(
            "Subject",
            placeholder="e.g. Biology, Physics, History"
        ).strip()

        chapter = st.text_input(
            "Chapter or Topic",
            placeholder="e.g. Cell Structure"
        ).strip()

    if st.button(
        "Generate Assessment",
        type="primary"
    ):
        if mode == "Verified Mathematics Demo":
            prepare_quiz(
                QUESTION_BANK.copy(),
                subject
            )
            st.success("Assessment ready!")

        elif not subject or not chapter:
            st.warning(
                "Please enter both subject and chapter."
            )

        else:
            try:
                with st.spinner(
                    "StudyLens is preparing your assessment..."
                ):
                    questions = generate_questions(
                        subject,
                        chapter,
                        count=8
                    )

                prepare_quiz(questions, subject)

                st.success(
                    f"Generated {len(questions)} questions."
                )

            except Exception as error:
                st.error(readable_ai_error(error))

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

                selected = st.radio(
                    "Choose your answer",
                    q["options"],
                    index=None,
                    key=f"{st.session_state.quiz_id}_q{i}"
                )

                confidence = st.select_slider(
                    "How confident are you?",
                    options=[
                        "Unsure",
                        "Somewhat Sure",
                        "Confident"
                    ],
                    value="Somewhat Sure",
                    key=f"{st.session_state.quiz_id}_c{i}"
                )

                selections.append(selected)
                confidences.append(confidence)

                st.divider()

            submitted = st.form_submit_button(
                "Submit Assessment",
                type="primary"
            )

        if submitted:
            if any(x is None for x in selections):
                st.warning(
                    "Please answer every question before submitting."
                )
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
            int(r["is_correct"]) for r in results
        )

        total = len(results)
        percentage = correct / total * 100

        st.subheader("Assessment Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("Your Score", f"{percentage:.1f}%")
        c2.metric("Correct Answers", correct)
        c3.metric("Total Questions", total)

        st.progress(percentage / 100)

        summary = analyse_results(results)

        st.subheader("Concept-Level Performance")

        st.dataframe(
            summary,
            hide_index=True,
            use_container_width=True
        )

        figure = px.bar(
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
            figure,
            use_container_width=True
        )

        st.subheader("Answer Review")

        for i, result in enumerate(results):
            icon = (
                "✅" if result["is_correct"]
                else "❌"
            )

            with st.expander(
                f"{icon} Question {i + 1}: {result['topic']}"
            ):
                st.write(result["question"])
                st.write(
                    "**Your answer:**",
                    result["selected"]
                )
                st.write(
                    "**Correct answer:**",
                    result["correct"]
                )
                st.write(
                    "**Explanation:**",
                    result["explanation"]
                )

        st.caption(
            "A short diagnostic cannot establish mastery "
            "with certainty. AI-generated answer keys "
            "should be checked for accuracy."
        )

        st.divider()

        if st.button(
            "✨ Generate Personalised AI Report",
            type="primary"
        ):
            try:
                with st.spinner(
                    "Analysing your learning gaps..."
                ):
                    report = create_learning_report(
                        st.session_state.quiz_subject,
                        results
                    )

                st.session_state.report = report

            except Exception as error:
                st.error(readable_ai_error(error))

        if st.session_state.report:
            st.subheader("Your Personalised Learning Report")
            st.markdown(st.session_state.report)


# ==========================================
# LEARNING DASHBOARD
# ==========================================

elif page == "Learning Dashboard":
    st.title("📊 Learning Analytics Dashboard")

    history = load_results(student)

    if history.empty:
        st.info(
            "Complete a diagnostic assessment "
            "to unlock your learning dashboard."
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
        ).round(1)

        summary["concept"] = (
            summary["subject"] + " — " + summary["topic"]
        )

        st.subheader("Concept Performance")

        figure = px.bar(
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
            figure,
            use_container_width=True
        )

        st.subheader("Revision Priorities")

        weak_topics = summary.sort_values(
            ["accuracy", "attempted"],
            ascending=[True, False]
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

        assessment_history = (
            assessment_history
            .sort_values(["created_at", "session_id"])
            .reset_index(drop=True)
        )

        assessment_history["attempt_number"] = range(
            1,
            len(assessment_history) + 1
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
            "Scores from different topics and difficulty "
            "levels are not directly comparable."
        )

        st.download_button(
            "📥 Download Learning Data",
            history.to_csv(index=False),
            file_name="studylens_learning_data.csv",
            mime="text/csv"
        )


# ==========================================
# AI LEARNING COACH
# ==========================================

elif page == "AI Learning Coach":
    st.title("🤖 StudyLens Learning Coach")

    history = load_results(student)

    if history.empty:
        st.info(
            "Complete an assessment first to activate "
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
            format_func=lambda item: (
                f"{item[0]} — {item[1]}"
            )
        )

        subject, topic = chosen

        action = st.radio(
            "What assistance do you need?",
            [
                "Explain this concept",
                "Create a 3-day revision plan",
                "Generate targeted practice questions"
            ]
        )

        if st.button(
            "Get Personalised Assistance",
            type="primary"
        ):
            try:
                with st.spinner(
                    "Preparing your learning guidance..."
                ):

                    if action == (
                        "Generate targeted practice questions"
                    ):
                        questions = generate_questions(
                            subject,
                            topic,
                            focus=topic,
                            count=5
                        )

                        prepare_quiz(questions, subject)

                        st.session_state.tutor_reply = (
                            "✅ Your targeted practice assessment "
                            "is ready. Open **Diagnostic Test** "
                            "from the sidebar to attempt it."
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

                        if action == "Explain this concept":
                            instruction = """
You are StudyLens AI, a patient learning coach.

Explain the selected concept at Class 11 level.
Use clear steps and worked examples.
Explain mistakes using the supplied answers.
End with three practice exercises.

Do not invent assessment results.
"""
                        else:
                            instruction = """
You are StudyLens AI, a learning planner.

Prepare a personalised three-day revision plan.

Include:
- Daily learning objectives
- Estimated study time
- Practice activities
- Revision exercises
- Final self-test

Base recommendations on supplied evidence.
Do not invent student scores.
"""

                        payload = {
                            "subject": subject,
                            "concept": topic,
                            "recent_answers": answer_data
                        }

                        st.session_state.tutor_reply = ai_request(
                            instruction,
                            json.dumps(payload, default=str)
                        )

            except Exception as error:
                st.error(readable_ai_error(error))

        if st.session_state.tutor_reply:
            st.markdown(st.session_state.tutor_reply)



elif page == "Exhibition Demo":
    st.title("🎪 The StudyLens Challenge")

    st.subheader(
        "Same Marks. Different Learning Gaps."
    )

    st.write("""
    Imagine two students who each score **60%**
    in a Mathematics assessment.

    Should they follow the same revision plan?

    **StudyLens demonstrates why the answer is no.**
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

        figure_a = px.bar(
            demo,
            x="Concept",
            y="Student A",
            range_y=[0, 100],
            color="Student A",
            color_continuous_scale="RdYlGn"
        )

        st.plotly_chart(
            figure_a,
            use_container_width=True
        )

        st.warning(
            "Revision priorities: Sequences, "
            "Functions and Sets."
        )

    with right:
        st.metric("Student B", "60%")

        figure_b = px.bar(
            demo,
            x="Concept",
            y="Student B",
            range_y=[0, 100],
            color="Student B",
            color_continuous_scale="RdYlGn"
        )

        st.plotly_chart(
            figure_b,
            use_container_width=True
        )

        st.warning(
            "Revision priorities: Algebra, "
            "Trigonometry and Sets."
        )

    st.divider()

    st.success("""
    Two students can receive identical marks
    while having completely different learning needs.

    StudyLens AI provides concept-level insights
    to support personalised revision.
    """)

    st.caption(
        "Illustrative results, assuming equal "
        "question weight across all five concepts."
    )



st.divider()

st.caption(
    "© 2026 StudyLens AI | "
    "Personalised Learning Intelligence Platform | "
    "Exhibition Prototype"
)

st.caption(
    "AI-generated educational content may contain "
    "errors and should be verified independently. "
    "Learning-gap predictions are preliminary."
)

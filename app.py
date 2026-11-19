
import streamlit as st
import joblib
import re
import nltk
import docx
import pypdf
import sqlite3
import pandas as pd
import io
import html

from pathlib import Path
from datetime import datetime
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet


# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .block-container {
        padding-top: 2rem;
    }

    h1, h2, h3 {
        color: #17365D;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 18px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }

    .stButton button {
        border-radius: 7px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# 2. PATHS AND MODEL
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "resume_screening_model.pkl"

loaded_objects = joblib.load(MODEL_PATH)

model = loaded_objects["model"]
tfidf = loaded_objects["tfidf"]

nltk.data.path.append(r"C:\Users\Shelv\nltk_data")

stop_words = set(stopwords.words("english"))


# --------------------------------------------------
# 3. DATABASE
# --------------------------------------------------

DB_PATH = BASE_DIR / "screening_history.db"


def create_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screening_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            email TEXT,
            phone TEXT,
            predicted_category TEXT,
            confidence REAL,
            job_match REAL,
            screening_date TEXT
        )
    """)

    connection.commit()
    connection.close()


create_database()


# --------------------------------------------------
# 4. TEXT CLEANING
# --------------------------------------------------

def clean_resume(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# --------------------------------------------------
# 5. RESUME EXTRACTION
# --------------------------------------------------

def extract_pdf_text(file):

    reader = pypdf.PdfReader(file)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""

    return text


def extract_word_text(file):

    document = docx.Document(file)

    text = ""

    for paragraph in document.paragraphs:
        text += paragraph.text + "\n"

    for table in document.tables:
        for row in table.rows:
            text += " ".join(
                cell.text for cell in row.cells
            ) + "\n"

    return text


# --------------------------------------------------
# 6. CANDIDATE INFORMATION
# --------------------------------------------------

def extract_candidate_information(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    candidate_name = "Not identified"

    if lines:
        candidate_name = lines[0][:100]

    email_match = re.search(
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        text
    )

    phone_match = re.search(
        r"(?:\+?\d{1,3}[\s-]?)?(?:\d[\s-]?){8,13}",
        text
    )

    email = email_match.group(0) if email_match else "Not found"

    phone = phone_match.group(0).strip() if phone_match else "Not found"

    education_keywords = [
        "degree",
        "diploma",
        "bachelor",
        "master",
        "phd",
        "matric",
        "certificate",
        "qualification",
        "university",
        "college"
    ]

    education = []

    for line in lines:

        if any(
            keyword in line.lower()
            for keyword in education_keywords
        ):
            education.append(line[:200])

    skills_list = [
        "python", "sql", "excel", "power bi",
        "tableau", "machine learning", "data analysis",
        "communication", "project management",
        "leadership", "java", "javascript",
        "accounting", "financial analysis",
        "statistics", "data visualization",
        "microsoft office", "html", "css",
        "customer service", "administration",
        "data entry", "management"
    ]

    found_skills = []

    text_lower = text.lower()

    for skill in skills_list:

        if skill in text_lower:
            found_skills.append(skill.title())

    return {
        "name": candidate_name,
        "email": email,
        "phone": phone,
        "education": education,
        "skills": found_skills
    }


# --------------------------------------------------
# 7. PREDICTION
# --------------------------------------------------

def predict_resume(text):

    cleaned_text = clean_resume(text)

    transformed_text = tfidf.transform([cleaned_text])

    probabilities = model.predict_proba(transformed_text)[0]

    classes = model.classes_

    sorted_indices = probabilities.argsort()[::-1]

    top_predictions = []

    for index in sorted_indices[:3]:

        top_predictions.append({
            "category": classes[index],
            "confidence": float(probabilities[index]) * 100
        })

    return top_predictions


# --------------------------------------------------
# 8. JOB DESCRIPTION MATCHING
# --------------------------------------------------

def calculate_job_match(resume_text, job_description):

    cleaned_resume = clean_resume(resume_text)

    cleaned_job = clean_resume(job_description)

    resume_vector = tfidf.transform([cleaned_resume])

    job_vector = tfidf.transform([cleaned_job])

    score = cosine_similarity(
        resume_vector,
        job_vector
    )[0][0]

    return float(score) * 100


# --------------------------------------------------
# 9. SAVE SCREENING HISTORY
# --------------------------------------------------

def save_screening(candidate, predictions, job_match):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO screening_history (
            candidate_name,
            email,
            phone,
            predicted_category,
            confidence,
            job_match,
            screening_date
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        candidate["name"],
        candidate["email"],
        candidate["phone"],
        predictions[0]["category"],
        predictions[0]["confidence"],
        job_match,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

    connection.commit()
    connection.close()


# --------------------------------------------------
# 10. READ HISTORY
# --------------------------------------------------

def get_history():

    connection = sqlite3.connect(DB_PATH)

    history = pd.read_sql_query(
        "SELECT * FROM screening_history ORDER BY id DESC",
        connection
    )

    connection.close()

    return history


# --------------------------------------------------
# 11. EXCEL REPORT
# --------------------------------------------------

def generate_excel_report(results):

    dataframe = pd.DataFrame(results)

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Screening Results"
        )

    output.seek(0)

    return output


# --------------------------------------------------
# 12. PDF REPORT
# --------------------------------------------------

def generate_pdf_report(results):

    output = io.BytesIO()

    document = SimpleDocTemplate(
        output,
        pagesize=A4
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Resume Screening Report",
            styles["Title"]
        )
    )

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M"),
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 20))

    table_data = [[
        "Candidate",
        "Category",
        "Confidence",
        "Job Match"
    ]]

    for result in results:

        table_data.append([
            result["Candidate"],
            result["Predicted Category"],
            f'{result["Confidence"]:.2f}%',
            (
                f'{result["Job Match"]:.2f}%'
                if result["Job Match"] is not None
                else "N/A"
            )
        ])

    table = Table(
        table_data,
        repeatRows=1
    )

    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17365D")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    elements.append(table)

    document.build(elements)

    output.seek(0)

    return output


# --------------------------------------------------
# 13. APPLICATION HEADER
# --------------------------------------------------

st.title("📄 Resume Screening System")

st.write(
    "An intelligent application for analysing resumes, "
    "predicting job categories and comparing candidates "
    "against job descriptions."
)

st.divider()


# --------------------------------------------------
# 14. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Dashboard",
        "Screen Resumes",
        "Screening History",
        "About"
    ]
)


# --------------------------------------------------
# 15. DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.header("Dashboard")

    history = get_history()

    total_screenings = len(history)

    if total_screenings > 0:

        average_confidence = history["confidence"].mean()

        categories = history["predicted_category"].nunique()

    else:

        average_confidence = 0

        categories = 0

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Screenings",
        total_screenings
    )

    col2.metric(
        "Average Confidence",
        f"{average_confidence:.1f}%"
    )

    col3.metric(
        "Categories Identified",
        categories
    )

    st.subheader("Recent Screening Activity")

    if not history.empty:

        st.dataframe(
            history.head(10),
            use_container_width=True
        )

    else:

        st.info(
            "No screening records yet. "
            "Go to Screen Resumes to begin."
        )


# --------------------------------------------------
# 16. SCREEN RESUMES
# --------------------------------------------------

elif page == "Screen Resumes":

    st.header("Screen Candidates")

    st.write(
        "Upload multiple PDF or Word resumes, "
        "or paste resume text below."
    )

    uploaded_files = st.file_uploader(
        "Upload resumes",
        type=["pdf", "docx"],
        accept_multiple_files=True
    )

    pasted_resume = st.text_area(
        "Or paste a resume here",
        height=200
    )

    st.subheader("Job Description")

    job_description = st.text_area(
        "Paste the job description (optional)",
        height=180
    )

    if st.button(
        "Analyse Resumes",
        type="primary"
    ):

        documents = []

        for uploaded_file in uploaded_files:

            try:

                if uploaded_file.name.lower().endswith(".pdf"):

                    resume_text = extract_pdf_text(uploaded_file)

                else:

                    resume_text = extract_word_text(uploaded_file)

                if resume_text.strip():

                    documents.append((
                        uploaded_file.name,
                        resume_text
                    ))

                else:

                    st.warning(
                        f"Could not extract text from {uploaded_file.name}"
                    )

            except Exception as error:

                st.error(
                    f"Error reading {uploaded_file.name}: {error}"
                )

        if pasted_resume.strip():

            documents.append((
                "Pasted Resume",
                pasted_resume
            ))

        if not documents:

            st.warning(
                "Please upload a resume or paste resume text."
            )

        else:

            results = []

            st.subheader("Analysis Results")

            progress = st.progress(0)

            for index, (filename, resume_text) in enumerate(documents):

                try:

                    candidate = extract_candidate_information(
                        resume_text
                    )

                    predictions = predict_resume(resume_text)

                    if job_description.strip():

                        job_match = calculate_job_match(
                            resume_text,
                            job_description
                        )

                    else:

                        job_match = None

                    save_screening(
                        candidate,
                        predictions,
                        job_match
                    )

                    results.append({
                        "Candidate": candidate["name"],
                        "Email": candidate["email"],
                        "Phone": candidate["phone"],
                        "Predicted Category": predictions[0]["category"],
                        "Confidence": predictions[0]["confidence"],
                        "Job Match": job_match,
                        "Filename": filename
                    })

                    with st.expander(
                        f"Candidate {index + 1}: {candidate['name']}"
                    ):

                        st.write("**File:**", filename)

                        st.write("**Email:**", candidate["email"])

                        st.write("**Phone:**", candidate["phone"])

                        st.write("**Education identified:**")

                        if candidate["education"]:

                            for item in candidate["education"]:

                                st.write("- " + item)

                        else:

                            st.write("No education information identified.")

                        st.write("**Skills identified:**")

                        if candidate["skills"]:

                            st.write(", ".join(candidate["skills"]))

                        else:

                            st.write("No listed skills identified.")

                        st.markdown("### Top 3 Predicted Categories")

                        for prediction in predictions:

                            st.write(
                                f"**{prediction['category']}** — "
                                f"{prediction['confidence']:.2f}% confidence"
                            )

                        if job_match is not None:

                            st.write(
                                f"**Job Description Similarity:** "
                                f"{job_match:.2f}%"
                            )

                    progress.progress(
                        (index + 1) / len(documents)
                    )

                except Exception as error:

                    st.error(
                        f"Could not analyse {filename}: {error}"
                    )

            if results:

                st.session_state["screening_results"] = results

                st.success(
                    f"Successfully analysed {len(results)} resume(s)."
                )

                st.subheader("Candidate Comparison")

                comparison = pd.DataFrame(results)

                if "Job Match" in comparison.columns:

                    comparison = comparison.sort_values(
                        by="Job Match",
                        ascending=False,
                        na_position="last"
                    )

                st.dataframe(
                    comparison,
                    use_container_width=True
                )

                st.download_button(
                    "Download Excel Report",
                    data=generate_excel_report(results),
                    file_name="resume_screening_report.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.download_button(
                    "Download PDF Report",
                    data=generate_pdf_report(results),
                    file_name="resume_screening_report.pdf",
                    mime="application/pdf"
                )


# --------------------------------------------------
# 17. SCREENING HISTORY
# --------------------------------------------------

elif page == "Screening History":

    st.header("Screening History")

    history = get_history()

    if not history.empty:

        st.dataframe(
            history,
            use_container_width=True
        )

        st.download_button(
            "Download History as Excel",
            data=generate_excel_report(
                history.drop(columns=["id"]).to_dict("records")
            ),
            file_name="screening_history.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:

        st.info("There are no saved screening records yet.")


# --------------------------------------------------
# 18. ABOUT
# --------------------------------------------------

elif page == "About":

    st.header("About This Application")

    st.write("""
    The Resume Screening System uses a trained machine learning
    model to predict the most likely job category for a resume.

    **Main features:**

    - Multiple resume uploads.
    - PDF and Word document extraction.
    - Machine learning category predictions.
    - Top three predicted categories.
    - Candidate information extraction.
    - Job description text similarity.
    - Screening history.
    - Excel and PDF report generation.

    **Important limitations:**

    Predictions and similarity scores are automated indicators,
    not proof of a candidate's suitability or ability.

    Candidate information extraction may be incomplete or inaccurate.
    Always review original resumes and make hiring decisions
    using fair, job-related criteria and human judgement.
    """)

    st.info(
        "Candidate information is stored locally in the project's "
        "screening_history.db file. Protect this file because it "
        "may contain personal information."
    )


# --------------------------------------------------
# 19. FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Resume Screening System | Machine Learning Project"
)

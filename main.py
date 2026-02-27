import streamlit as st
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq
import re
from dotenv import load_dotenv
import os
import numpy as np

# ---------------- Load environment variables ----------------
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ---------------- Session States ----------------
if "form_submitted" not in st.session_state:
    st.session_state.form_submitted = False
if "resume" not in st.session_state:
    st.session_state.resume = ""
if "job_desc" not in st.session_state:
    st.session_state.job_desc = ""
if "ats_score" not in st.session_state:
    st.session_state.ats_score = 0.0
if "report" not in st.session_state:
    st.session_state.report = ""
if "avg_score" not in st.session_state:
    st.session_state.avg_score = 0.0

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Resume Analyzer 📝",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🛠 Resume Analyzer")
    st.markdown("""
    Welcome! This tool will help you:
    - Upload your **Resume** (PDF)
    - Paste the **Job Description**
    - Get **similarity scores** and **AI analysis**
    """)
    st.image(
        "https://www.freepik.com/free-photos-vectors/get-a-job",
        use_column_width=True
    )
    st.markdown("---")

# ---------------- Header ----------------
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>Resume Analyzer 📝</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Compare your Resume with a Job Description and get AI-generated insights!</p>", unsafe_allow_html=True)

# ---------------- Functions ----------------
def extract_pdf_text(uploaded_file):
    try:
        return extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Error extracting text: {str(e)}")
        return "Could not extract text."

def calculate_similarity_bert(text1, text2):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings1 = model.encode([text1])
    embeddings2 = model.encode([text2])
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity)  

def get_report(resume, job_desc):
    client = Groq(api_key=api_key)
    prompt=f"""
    # Context:
    - You are an AI Resume Analyzer, you will be given Candidate's resume and Job Description of the role he is applying for.

    # Instruction:
    - Analyze candidate's resume based on the possible points that can be extracted from job description,and give your evaluation on each point with the criteria below:  
    - Consider all points like required skills, experience,etc that are needed for the job role.
    - Calculate the score to be given (out of 5) for every point based on evaluation at the beginning of each point with a detailed explanation.  
    - If the resume aligns with the job description point, mark it with ✅ and provide a detailed explanation.  
    - If the resume doesn't align with the job description point, mark it with ❌ and provide a reason for it.  
    - If a clear conclusion cannot be made, use a ⚠️ sign with a reason.  
    - The Final Heading should be "Suggestions to improve your resume:" and give where and what the candidate can improve to be selected for that job role.

    # Inputs:
    Candidate Resume: {resume}
    ---
    Job Description: {job_desc}

    # Output:
    - Each any every point should be given a score (example: 3/5 ). 
    - Mention the scores and relevant emoji at the beginning of each point and then explain the reason.
    """
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat_completion.choices[0].message.content

def extract_scores(text):
    pattern = r'(\d+(?:\.\d+)?)/5'
    matches = re.findall(pattern, text)
    scores = [float(match) for match in matches]
    return scores

# ---------------- Tabs ----------------
tab1, tab2 = st.tabs(["📄 Resume Upload", "📊 Analysis Results"])

# ---------------- Tab 1: Resume Upload ----------------
with tab1:
    st.subheader("Upload Resume & Job Description")
    col1, col2 = st.columns([1,2])
    with col1:
        resume_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")
    with col2:
        st.session_state.job_desc = st.text_area("💼 Paste Job Description", placeholder="Enter the job description here...")

    if st.button("Analyze Resume"):
        if resume_file and st.session_state.job_desc.strip() != "":
            st.info("Extracting Resume text...")
            st.session_state.resume = extract_pdf_text(resume_file)

            with st.spinner("Calculating similarity score..."):
                st.session_state.ats_score = calculate_similarity_bert(st.session_state.resume, st.session_state.job_desc)

            with st.spinner("Generating AI Analysis Report..."):
                st.session_state.report = get_report(st.session_state.resume, st.session_state.job_desc)
                scores = extract_scores(st.session_state.report)
                st.session_state.avg_score = sum(scores) / (5*len(scores)) if scores else 0.0

            st.session_state.form_submitted = True
            st.success("Analysis Complete! Check the Analysis Results tab.")
        else:
            st.warning("Please provide both Resume and Job Description!")

# ---------------- Tab 2: Analysis Results ----------------
with tab2:
    if st.session_state.form_submitted:
        st.markdown("### 🧮 Scores")
        col1, col2 = st.columns(2)
        col1.metric(label="ATS Similarity Score", value=f"{st.session_state.ats_score:.2f}", delta="0-1 scale")
        col2.metric(label="Average AI Score", value=f"{st.session_state.avg_score:.2f}", delta="0-5 scale")

        # Progress bar (0-100%)
        st.progress(int(float(st.session_state.ats_score) * 100))

        st.markdown("### 🧾 Generated Analysis Report")
        st.markdown(
            f"<div style='padding: 15px; background-color: #901838; border-radius: 10px;'>{st.session_state.report}</div>",
            unsafe_allow_html=True
        )

        st.download_button(
            label="📥 Download Report",
            data=st.session_state.report,
            file_name="Resume_Report.txt",
            mime="text/plain"
        )
    else:
        st.info("Analysis results will appear here after you submit the Resume & Job Description in the first tab.")

# ---------------- Footer ----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color: gray;'>Author: Mk Ador, 2026</p>", unsafe_allow_html=True)
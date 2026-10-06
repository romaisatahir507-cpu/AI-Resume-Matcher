import os
import requests
import re
import hashlib

import streamlit as st

from database.db import SessionLocal
from database.models import Candidate, Resume

from services.resume_parser import extract_resume_text
from services.text_cleaner import clean_resume_text
from services.llm import (
    analyze_resume, 
    match_resume_with_job, 
    analyze_resume_with_rag,
    generate_llm_response
)
from services.tfidf_matcher import calculate_tfidf_match
from services.embeddings import calculate_semantic_similarity
from services.scoring_engine import calculate_final_score, get_recommendation
from services.ml_matcher import predict_ml_match
from services.chroma_service import add_resume_to_chroma
from services.rag_service import retrieve_resume_context
from services.api_client import match_resume



API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title = "AI Resume Matcher",
                   page_icon = "📄",
                   layout = "wide"

)


st.markdown("""
<style>

/* =========================================================
   GLOBAL THEME
   ========================================================= */

.stApp {
    background-color: #e5e7eb;
    color: #1f2937;
    font-family: "Segoe UI", Arial, sans-serif;
}


/* =========================================================
   MAIN CONTENT AREA
   ========================================================= */

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* =========================================================
   MAIN APPLICATION TITLE
   ========================================================= */

h1 {
    color: #172554;
    font-size: 2.6rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.5px;
    margin-bottom: 0.3rem !important;
}


/* =========================================================
   MAJOR SECTION HEADINGS
   ========================================================= */

h2 {
    color: #1e3a5f;
    font-size: 1.8rem !important;
    font-weight: 650 !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
}


/* =========================================================
   SUBSECTION HEADINGS
   ========================================================= */

h3 {
    color: #334155;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 0.7rem !important;
}


/* =========================================================
   NORMAL TEXT
   ========================================================= */

p {
    color: #475569;
    font-size: 0.98rem;
    line-height: 1.6;
}


/* =========================================================
   INPUT LABELS
   ========================================================= */

label {
    color: #334155 !important;
    font-weight: 600 !important;
}


/* =========================================================
   TEXT INPUTS / TEXT AREAS
   ========================================================= */

.stTextInput input,
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1f2937 !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.75rem !important;
}


/* Input focus */

.stTextInput input:focus,
.stTextArea textarea:focus {
    border-color: #64748b !important;
    box-shadow: 0 0 0 1px #64748b !important;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 10px;
    padding: 0.5rem;
}


/* =========================================================
   BUTTONS
   ========================================================= */

.stButton > button {
    background-color: #4f7cac;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 0.55rem 1.2rem;
    font-weight: 600;
    transition: all 0.2s ease;
}


.stButton > button:hover {
    background-color: #3f6d99;
    color: #ffffff;
    border: none;
    transform: translateY(-1px);
}


/* =========================================================
   METRIC CARDS
   ========================================================= */

[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 1rem;
}


/* =========================================================
   DATAFRAMES / TABLES
   ========================================================= */

[data-testid="stDataFrame"] {
    background-color: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.25rem;
}


/* =========================================================
   ALERT / MESSAGE BOXES
   ========================================================= */

[data-testid="stAlert"] {
    border-radius: 8px;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background-color: #eef2f7;
    border-right: 1px solid #dbe2ea;
}


[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #1e3a5f !important;
}


/* =========================================================
   SELECTBOX / MULTISELECT
   ========================================================= */

[data-baseweb="select"] > div {
    background-color: #ffffff;
    border-radius: 8px;
    border-color: #cbd5e1;
}


/* =========================================================
   DIVIDERS
   ========================================================= */

hr {
    border: none;
    border-top: 1px solid #dbe2ea;
    margin: 1.5rem 0;
}


/* =========================================================
   HIDE STREAMLIT HEADING ANCHOR / LINK ICONS
   ========================================================= */

[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stHeaderActionElements"] svg {
    display: none !important;
}


/* =========================================================
   GENERAL SPACING
   ========================================================= */

[data-testid="stVerticalBlock"] {
    gap: 0.6rem;
}


/* =========================================================
   RESULT CARDS
   ========================================================= */

.result-card {
    background-color: #ffffff;
    border: 1px solid #d8dee6;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}


/* =========================================================
   AI RESPONSE CARD
   ========================================================= */

.ai-response-card {
    background-color: #ffffff;
    border-left: 4px solid #4f7cac;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    margin-top: 0.8rem;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}


/* =========================================================
   CANDIDATE CARD
   ========================================================= */

.candidate-card {
    background-color: #ffffff;
    border: 1px solid #d8dee6;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.8rem 0 1.2rem 0;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}


/* =========================================================
   SCORE VALUE
   ========================================================= */

.score-value {
    font-size: 1.35rem;
    font-weight: 700;
    color: #3f6d99;
}


/* =========================================================
   CANDIDATE NAME
   ========================================================= */

.candidate-name {
    color: #1e3a5f;
    font-size: 1.25rem;
    font-weight: 650;
    margin-bottom: 0.8rem;
}


/* =========================================================
   SMALL INFORMATION TEXT
   ========================================================= */

.info-text {
    color: #64748b;
    font-size: 0.9rem;
}


/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #eef2f7;
    border-right: 1px solid #d8dee6;
}

[data-testid="stSidebar"] h2 {
    color: #1e3a5f !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    margin-bottom: 0.2rem !important;
}

[data-testid="stSidebar"] p {
    color: #64748b;
    font-size: 0.9rem;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    margin-top: 0.5rem;
}

/* UPLOAD SECTION */
[data-testid="stFileUploader"] section {
    background-color: #ffffff;
    border-radius: 10px;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px dashed #94a3b8 !important;
    border-radius: 10px !important;
    background-color: #f8fafc !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: #4f7cac !important;
    background-color: #f1f5f9 !important;
}


/* AI RESULT CARDS */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border-radius: 12px;
    border: 1px solid #d8dee6;
    box-shadow: 0 2px 6px rgba(15, 23, 42, 0.05);
}


/* MATCH SCORE METRICS */
[data-testid="stMetricValue"] {
    font-size: 1.15rem !important;
    font-weight: 650 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.9rem !important;
    font-weight: 600 !important;
}

/* COMPARISON TABLE HEADER */
[data-testid="stDataFrame"] [role="columnheader"] {
    color: #1e293b !important;
    font-weight: 700 !important;
}

</style>
""", unsafe_allow_html=True)



#----------------------
# FastAPI Status
#----------------------

with st.sidebar:
    st.header("System Status")
    st.caption("AI Resume Matcher Services")

    if st.button("Check API Status", use_container_width = True):
        try:
            response = requests.get(
                f"{API_URL}/docs",
                timeout = 5
            )

            if response.status_code == 200:
                st.success("FastAPI is running")
            else:
                st.error("FastAPI is not responding")
        
        except requests.exceptions.RequestException:
            st.error("FastAPI is not running")


st.title("AI Resume Matcher")
st.markdown(
    "AI-powered resume screening, matching, ranking, and candidate analysis."
)

st.header("Resume Management")
st.subheader("Upload Candidate Resume")

st.caption("Enter candidate details and upload a PDF or DOCX resume for analysis.")

col1, col2 = st.columns(2)

with col1:
    candidate_name = st.text_input("Candidate Name", placeholder = "Enter candidate name")
    candidate_email = st.text_input("Candidate Email", placeholder = "Enter candidate email")

with col2:
    candidate_phone = st.text_input("Candidate Phone", placeholder = "Enter candidate phone")
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"], help = "Supported formats: PDF and DOCX")



# Initialize Streamlit session state

if "cleaned_text" not in st.session_state:
    st.session_state.cleaned_text = None

if "resume_text" not in st.session_state:
    st.session_state.resume_text = None

if "candidate_id" not in st.session_state:
    st.session_state.candidate_id = None

if "resume_id" not in st.session_state:
    st.session_state.resume_id = None

if "resume_analysis" not in st.session_state:
    st.session_state.resume_analysis = None

if "job_match_analysis" not in st.session_state:
    st.session_state.job_match_analysis = None


#-----------------------------
# Hashing Function
#-----------------------------


def generate_resume_hash(resume_text):
    normalized_text = " ".join(resume_text.lower().split())

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()



#--------------------------
# Upload Resume Button
#--------------------------



if st.button("Upload Resume", use_container_width = True):

    if not candidate_name:

        st.error("Please enter the candidate name.")

    elif uploaded_file is None:

            st.error("Please upload a resume file.")

    else:

        try:
                #----------------------#
                # Save Uploaded File 
                #----------------------#

                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())


                #----------------------#
                # Extract Resume Text
                #----------------------#

                resume_text = extract_resume_text(file_path)

                if not resume_text:

                    st.error("Could not extract text from the uploaded resume.")

                else:

                    #----------------------#
                    # Clean Resume Text
                    #----------------------#

                    cleaned_text = clean_resume_text(resume_text)

                    # store resume data in session state

                    st.session_state.resume_text = resume_text
                    st.session_state.cleaned_text = cleaned_text

                    #------------------------------
                    # Check for Duplicate Resume
                    #------------------------------

                    resume_hash = generate_resume_hash(cleaned_text)

                    db = SessionLocal()

                    existing_resume = (
                        db.query(Resume).filter(Resume.resume_hash == resume_hash).first()
                    )

                    if existing_resume:

                        existing_candidate = (
                            db.query(Candidate).filter(Candidate.id == existing_resume.candidate_id).first()
                        )

                        candidate_name_existing = (existing_candidate.name if existing_candidate else "Unknown")

                        db.close()

                        st.warning(
                            f"This resume already exists in the database. "
                            f"Candidate: {candidate_name_existing}, "
                            f"Resume ID: {existing_resume.id}"
                        )

                        st.stop()

                    #----------------------#
                    # Analyze Resume with LLM
                    #----------------------#

                    resume_analysis = analyze_resume(cleaned_text)

                    if not resume_analysis:
                        st.error("Could not analyze resume using AI.")

                    else:
                        # Store LLM analysis is session state

                        st.session_state.resume_analysis = resume_analysis


                        #----------------------#
                        # Create Candidate 
                        #----------------------#

                        candidate = Candidate(name=candidate_name, email=candidate_email, phone=candidate_phone)

                        db.add(candidate)
                        db.commit()
                        db.refresh(candidate)


                        #----------------------#
                        # Create Resume Record
                        #----------------------#

                        resume = Resume(
                            candidate_id = candidate.id,
                            filename = uploaded_file.name,
                            file_path = file_path,
                            raw_text = cleaned_text,
                            ai_analysis = resume_analysis,
                            resume_hash = resume_hash
                        )
                    
                        db.add(resume)
                        db.commit()
                        db.refresh(resume)

                        add_resume_to_chroma(
                            resume_id=resume.id,
                            resume_text=cleaned_text
                        )
                        

                        #---------------------#
                        # Store IDs
                        #---------------------#
 
                        st.session_state.candidate_id = candidate.id
                        st.session_state.resume_id = resume.id

                        #----------------------#
                        # Close Database
                        #----------------------#

                        db.close()

                        #-----------------------#
                        # 9. Success Message
                        #-----------------------#


                        st.success("Resume uploaded and information stored successfully.")

        
        except Exception as e:
            
            st.error(f"An Error Occurred:{str(e)}")


#--------------------------------
# Display AI Resume Overview outside Upload Button
#--------------------------------

if st.session_state.resume_analysis:
    
    st.header("Resume Analysis")
    st.subheader("AI Resume Overview")

    with st.container(border=True):
        st.markdown(st.session_state.resume_analysis.replace("<br>", "\n"))


#---------------------------
# Display AI Job Match Analysis
#--------------------------

if st.session_state.job_match_analysis:

    st.header("Job Matching")
    st.subheader("AI Job Match Analysis")
    
    with st.container(border=True):
        st.markdown(st.session_state.job_match_analysis)



#---------------------#
# Session State Initialization
#---------------------#

if "tfidf_score" not in st.session_state:
    st.session_state.tfidf_score = None

if "semantic_score" not in st.session_state:
    st.session_state.semantic_score = None

if "final_score" not in st.session_state:
    st.session_state.final_score = None

if "ml_score" not in st.session_state:
    st.session_state.ml_score = None

if "rag_context" not in st.session_state:
    st.session_state.rag_context = ""



#----------------------#
# Job Description
#----------------------#

job_description = ""

if st.session_state.cleaned_text:
    
    st.header("Job Matching")
    st.subheader("Job Description")

    job_description = st.text_area("Paste the job description here:", height = 200)


#----------------------#
# Resume Job Matching
#----------------------#

if st.button("Match Resume with Job"):
    
    if not job_description.strip():

        st.warning("Please enter a job description.")

    elif not st.session_state.cleaned_text:

        st.warning("Please upload a resume first.")

    else:

        try:

            with st.spinner("Matching resume with Job..."):

                api_result = match_resume(
                    st.session_state.resume_id,
                    job_description
                )

                st.success("FastAPI matching successful.")

                st.write(api_result)

        except requests.exceptions.ConnectionError:

            st.error(
                "Unable to connect to the matching service."
                "Please make sure FastAPI is running and try again."
            )


        except Exception as e:

            st.error(f"Matching Error: {str(e)}")



#-----------------------
# Candidate Ranking Dashboard
#-----------------------


st.header("Candidate Ranking Dashboard")

ranking_job_description = st.text_area("Enter Job Description for Candidate Ranking", height = 200)

if st.button("Rank Candidates"):

    if not ranking_job_description.strip():

        st.warning("Please enter a job description.")

    else:

        try:

            db = SessionLocal()

            # Get all resumes
            resumes = (
                db.query(Resume).order_by(Resume.id.desc()).all()
            )

            st.write(f"Total resumes found: {len(resumes)}")

            if not resumes:

                st.warning("No resumes found in database.")

            else:

                # Load all candidates once
                candidates = db.query(Candidate).all()

                candidate_lookup = {
                    candidate.id: candidate.name
                    for candidate in candidates
                }

                ranking_results = []

                # Progress bar
                progress_bar = st.progress(0)

                # Status message
                status_text = st.empty()

                total_resumes = len(resumes)

                for index, resume in enumerate(resumes):

                    candidate_name = candidate_lookup.get(
                        resume.candidate_id,
                        "Unknown"
                    )

                    status_text.write(
                        f"Processing {index + 1} of {total_resumes}: "
                        f"{candidate_name}"
                    )

                    try:

                        api_result = match_resume(
                            resume.id,
                            ranking_job_description
                        )

                        ranking_results.append({
                            "Candidate": candidate_name,
                            "Resume ID": resume.id,
                            "TF-IDF Score": api_result.get("tfidf_score", 0),
                            "Semantic Score": api_result.get("semantic_score", 0),
                            "Final Score": api_result.get("final_score", 0),
                            "Recommendation": api_result.get("recommendation", "N/A")
                        })

                    except Exception as e:

                        st.warning(
                            f"Could not process Resume ID "
                            f"{resume.id}: {str(e)}"
                        )

                    # Update progress 
                    progress_bar.progress(
                        (index + 1) / total_resumes
                    )

                db.close()

                status_text.empty()

                # Sort by final score
                ranking_results = sorted(
                    ranking_results, 
                    key = lambda x: x["Final Score"],
                    reverse = True
                )

                # Assign ranks
                for rank, result in enumerate(ranking_results, start = 1):
                    result["Rank"] = rank

                st.subheader("Ranking Results")

                st.caption(
                    "Candidates ranked according to the calculated matching scores."
                )

                ranking_table = []

                for result in ranking_results:

                    ranking_table.append({
                        "Rank": result["Rank"],
                        "Candidate": result["Candidate"],
                        "TF-IDF Score": result["TF-IDF Score"],
                        "Semantic Score": result["Semantic Score"],
                        "Final Score": result["Final Score"],
                        "Recommendation": result["Recommendation"]
                    })

                st.dataframe(
                    ranking_table,
                    width = "stretch",
                    hide_index = True
                )

                st.success(
                    f"Ranking completed successfully for "
                    f"{len(ranking_results)} candidates."
                )

        except Exception as e:

            st.error(f"Ranking Error: {str(e)}")



#------------------------
# AI Recruiter Chatbot
#------------------------

st.header("AI Recruiter Chatbot")

recruiter_question = st.text_input(
    "Recruiter Question",
    placeholder="Ask about candidate skills, experience, education, or projects..."
)

if st.button("Ask Recruiter AI"):

    if not recruiter_question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner("AI Recruiter is thinking..."):

                # Retrieve resume context from ChromaDB

                recruiter_context = retrieve_resume_context(
                    recruiter_question,
                    n_results = 1
                )

                # Create recruiter prompt
                recruiter_prompt = f"""

                You are an AI recruiter assistant.

                Answer the recruiter's question using ONLY the candidate information provided
                in the retrieved resume context.

                Do not invent information.
                If the information is not available, say: 
                "Not mentioned in the resume."

                Candidate Resume Context:
                --------------------------
                {recruiter_context}
                --------------------------

                Recruiter question:
                {recruiter_question}

                Give a clear, concise, and professional answer.
                """

                # Generate answer using Groq
                recruiter_answer = generate_llm_response(recruiter_prompt)

                st.subheader("Recruiter AI Response")

                with st.container(border=True):
                    st.markdown(recruiter_answer)


        except Exception as e:

            st.error(f"Recruiter Chatbot Error: {str(e)}")


#------------------------
# Candidate Comparison
#------------------------


if "comparison_results" not in st.session_state:
    st.session_state.comparison_results = []

st.header("Candidate Comparison")

# Shared Job Description
job_description_compare = st.text_area(
    "Job Description",
    height = 180,
    placeholder = "Paste the job description here..."
)

# Get all candidates
db = SessionLocal()

try:

    candidates = db.query(Candidate).all()

    if len(candidates) < 2:

        st.info("At least 2 candidates are required for comparison.")

    else:

        candidate_options = {
            f"{candidate.name} (ID: {candidate.id})": candidate.id for candidate in candidates
        }

        selected_candidates = st.multiselect(
            "Select candidates to compare",
            options = list(candidate_options.keys()),
            max_selections = 3
        )

        if len(selected_candidates) < 2:

            st.info("Please select at least 2 candidates.")

        elif not job_description_compare.strip():

            st.warning("Please enter a job description.")

        else:

            if st.button(
                "Compare Candidates",
                key = "compare_candidates"
            ):
                comparison_results = []

                for selected_candidate in selected_candidates:

                    candidate_id = candidate_options[selected_candidate]

                    candidate = db.query(Candidate).filter(
                        Candidate.id == candidate_id
                    ).first()

                    resume = db.query(Resume).filter(
                        Resume.candidate_id == candidate_id
                    ).order_by(Resume.id.desc()).first()

                    if not resume:

                        st.warning(f"No resume found for {candidate.name}")

                        continue

                    try:

                        response = match_resume(resume.id, job_description_compare)

                        comparison_results.append({
                            "Candidate": candidate.name,
                            "Candidate ID": candidate.id,
                            "Resume ID": resume.id,
                            "TF-IDF Score": response.get("tfidf_score", 0),
                            "Semantic Score": response.get("semantic_score", 0),
                            "ML Score": response.get("ml_score", 0),
                            "Final Score": response.get("final_score", 0),
                            "Recommendation": response.get("recommendation", "N/A")
                        })


                    except Exception as e:

                        st.error(
                            f"Could not process."
                            f"{candidate.name}: {e}"
                        )

                # Save results
                st.session_state.comparison_results = comparison_results

            
                #-------------------------
                # Comparison Results
                #-------------------------

            if st.session_state.comparison_results:

                comparison_results = st.session_state.comparison_results

                st.subheader("Comparison Results")

                st.caption("Side-by-side matching scores for the selected candidates.")

                comparison_display = []

                for result in comparison_results:

                    comparison_display.append({
                        "Candidate": result["Candidate"],
                        "TF-IDF": f"{result['TF-IDF Score']:.2f}%",
                        "Semantic": f"{result['Semantic Score']:.2f}%",
                        "ML": f"{result['ML Score']:.2f}%",
                        "Final Score": f"{result['Final Score']:.2f}%",
                        "Recommendation": result["Recommendation"]
                    })

                st.dataframe(
                    comparison_display,
                    use_container_width = True,
                    hide_index = True
                )


                #----------------------------
                # Individual Candidate Scores
                #----------------------------

                st.subheader("Candidate Match Scores")

                for result in comparison_results:

                    st.markdown(
                        f'<div class="candidate-name">{result["Candidate"]}</div>',
                        unsafe_allow_html = True
                    )

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                            
                        st.metric("TF-IDF", f"{result['TF-IDF Score']:.2f}%")

                    with col2:

                        st.metric("Semantic", f"{result['Semantic Score']:.2f}%")

                    with col3:

                        st.metric("ML", f"{result['ML Score']:.2f}%")

                    with col4:

                        st.metric("Final Score", f"{result['Final Score']:.2f}%")

                    st.markdown(
                        f'<div class="info-text">Recommendation: '
                        f'<strong>{result["Recommendation"]}</strong></div>',
                        unsafe_allow_html=True
                    )
                    

                    
                #-----------------------------
                # AI Candidate Comparison
                #-----------------------------

                st.subheader("AI-Powered Candidate Comparison")

                if st.button(
                    "Generate AI comparison",
                    key = "generate_ai_comparison"
                ):
                    comparison_prompt = """
                    You are an AI recruiter.

                    Compare the following candidates for the same job description.

                    JOB DESCRIPTION:
                    """

                    comparison_prompt += f"""

                    {job_description_compare}

                    CANDIDATES:

                    """

                    for result in comparison_results:

                        candidate = db.query(Candidate).filter(
                            Candidate.id == result["Candidate ID"]
                        ).first()

                        resume = db.query(Resume).filter(
                            Resume.id == result["Resume ID"]
                        ).first()

                        resume_text = ""

                        if resume:
                            resume_text = resume.raw_text

                        comparison_prompt += f"""

                        ------------------------------------------
                        Candidate: {result['Candidate']}
                        ------------------------------------------

                        Resume Context:
                        {resume_text}

                        Matching Score:
                        TF-IDF Scores: {result['TF-IDF Score']:.2f}%
                        Semantic Score: {result['Semantic Score']:.2f}%
                        ML Score: {result['ML Score']:.2f}%
                        Final Score: {result['Final Score']:.2f}%
                        Recommendation: {result['Recommendation']}

                        """

                    comparison_prompt += """

                    Compare the candidates using only the information provided.

                    For EACH candidate, provide:

                    1. Candidate Summary - maximum 2 sentences
                    2. Technical Skills - concise bullet list
                    3. Soft Skills - concise bullet list
                    4. Education - concise bullet list
                    5. Work Experience - list the relevant roles and summarize in 1 sentence
                    6. Projects - concise bullet list
                    7. Certifications - concise bullet list
                    8. Experience Level - 1 sentence
                    9. Relevant Strengths - maximum 3 bullets
                    10. Potential Skill Gaps - maximum 3 bullets
                    11. Relevant Skills Matching the Job - maximum 5 bullets
                    12. Overall Match with the Job - 1-2 sentences

                    Keep the information concise. Do not repeat the same information across sections.

                    Then provide a section called:

                    ## Candidate Comparison

                    Compare the candidates based on:

                    - Technical skills
                    - Relevant experience
                    - Education
                    - Projects
                    - Job-relevant skills
                    - Skill gaps
                    - Matching scores

                    Keep this comparison concise. Use bullet points, not tables.

                    Finally provide:

                    Recruiter Summary

                    Give a concise factual summary of the key differences between the candidates for this job.

                    Do not make a hiring decision or recommend who should be hired.
                    Do not rank candidates beyond reporting the supplied matching scores.
                    Do not use phrases such as "strongest candidate", "best candidate",
                    "prioritize", "hire", or "not viable".

                    Do not invent information.
                    
                    Do not infer, assume, or derive skills, tools, certifications,
                    experience, soft skills, or achievements that are not explicitly
                    stated in the resume.
 
                    Do not treat something as present merely because it is related to
                    another skill. For example, do not assume FAISS experience from
                    ChromaDB experience.

                    If information is not explicitly provided, state:
                    "Not mentioned in resume."
                    """

                    try:
                        ai_response = analyze_resume(comparison_prompt)

                        ai_response = re.sub(
                             r"^#{1,6}\s+(.+)$",
                            r"**\1**",
                            ai_response,
                            flags = re.MULTILINE
                        )

                        st.markdown(ai_response)

                    except Exception as e:

                        st.error(f"AI comparison failed: {e}")



finally:
    db.close()
                            

#----------------------#
# Display Resume Information
#----------------------#

if st.session_state.cleaned_text:

    st.header("Resume Information")
    st.subheader("Cleaned Resume Text")

    st.caption("Normalized resume text extracted and processed by the application.")
    st.text_area("Resume Content", st.session_state.cleaned_text, height=400)
    st.write(f"Candidate ID: {st.session_state.candidate_id}")
    st.write(f"Resume ID: {st.session_state.resume_id}")

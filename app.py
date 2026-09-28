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



UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title = "AI Resume Matcher",
                   page_icon = "📄",
                   layout = "wide"

)


st.markdown("""
<style>
/* Hide Streamlit heading anchor/link icons */
[data-testid="stHeaderActionElements"] {
    display: none !important;
}

[data-testid="stHeaderActionElements"] svg {
    display: none !important;
}
</style>
""",unsafe_allow_html = True)



#----------------------
# FastAPI Status
#----------------------

with st.sidebar:
    st.header("System")
    if st.button("Check API Status"):
        try:
            response = requests.get(
                "http://127.0.0.1:8000/docs",
                timeout = 5
            )

            if response.status_code == 200:
                st.success("FastAPI is running")
            else:
                st.error("FastAPI is not responding")
        
        except requests.exceptions.RequestException:
            st.error("FastAPI is not running")


st.title("AI Resume Matcher")
st.write("Upload a candidate resume to extract and store its information.")
st.header("Upload Candidate Resume")

candidate_name = st.text_input("Candidate Name")
candidate_email = st.text_input("Candidate Email")
candidate_phone = st.text_input("Candidate Phone")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx"])


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



if st.button("Upload Resume"):

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
    
    st.subheader("AI Resume Overview")

    st.markdown(st.session_state.resume_analysis)


#---------------------------
# Display AI Job Match Analysis
#--------------------------

if st.session_state.job_match_analysis:

    st.subheader("AI Job Match Analysis")

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

            resumes = db.query(Resume).order_by(Resume.id.desc()).all()
            st.write(f"Total resumes found: {len(resumes)}")

            if not resumes:

                st.warning("No resumes found in database.")

            else:

                ranking_results = []

                with st.spinner("Ranking candidates..."):

                    for resume in resumes:

                        try:

                            api_result = match_resume(resume.id, ranking_job_description)

                            candidate = (db.query(Candidate).filter(Candidate.id == resume.candidate_id).first())

                            ranking_results.append({
                                "Candidate": candidate.name if candidate else "Unkonown",
                                "Resume ID": resume.id,
                                "TF-IDF Score": api_result.get("tfidf_score", 0),
                                "Semantic Score": api_result.get("semantic_score", 0),
                                "Final Score": api_result.get("final_score", 0),
                                "Recommendation": api_result.get("recommendation", "N/A")
                            })

                        except Exception as e:

                            st.warning(f"Could not process Resume ID {resume.id}: {str(e)}")

                db.close()

                # Sort candidates by final score
                ranking_results = sorted(
                    ranking_results,
                    key=lambda x: x["Final Score"],
                    reverse=True
                )

                # Add ranking position

                for rank, result in enumerate(ranking_results, start=1):
                    result["Rank"] = rank

                st.subheader("Candidate Rankings")

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
                    use_container_width=True,
                    hide_index=True
                )


        except Exception as e:

            st.error(f"Ranking Error: {str(e)}")



#------------------------
# AI Recruiter Chatbot
#------------------------

st.header("AI Recruiter Chatbot")

recruiter_question = st.text_input("Ask a question about the candidate")

if st.button("Ask Recruiter AI"):

    if not recruiter_question.strip():

        st.warning("Please enter a question.")

    elif not st.session_state.resume_id:

        st.warning("Please upload a resume first.")

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

                st.subheader("AI Recruiter Response")

                st.write(recruiter_answer)


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

                st.subheader("Candidate Scores")

                for result in comparison_results:

                    st.markdown(f"### {result['Candidate']}")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                            
                        st.metric("TF-IDF", f"{result['TF-IDF Score']:.2f}%")

                    with col2:

                        st.metric("Semantic", f"{result['Semantic Score']:.2f}%")

                    with col3:

                        st.metric("ML", f"{result['ML Score']:.2f}%")

                    with col4:

                        st.metric("Final Score", f"{result['Final Score']:.2f}%")

                    st.write(
                        f"Recommendation: "
                        f"**{result['Recommendation']}**"
                    )

                    
                #-----------------------------
                # AI Candidate Comparison
                #-----------------------------

                st.subheader("AI Candidate Comparison")

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

    st.subheader("Cleaned Resume Text.")
    st.text_area("Resume Content", st.session_state.cleaned_text, height=400)
    st.write(f"Candidate ID: {st.session_state.candidate_id}")
    st.write(f"Resume ID: {st.session_state.resume_id}")

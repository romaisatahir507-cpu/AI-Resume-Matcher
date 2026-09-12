import os
import requests

import streamlit as st

from database.db import SessionLocal
from database.models import Candidate, Resume

from services.resume_parser import extract_resume_text
from services.text_cleaner import clean_resume_text
from services.llm import (
    analyze_resume, 
    match_resume_with_job, 
    analyze_resume_with_rag
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
        
        except requests.expectations.RequestException:
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
                        # Create Database Session
                        #----------------------#

                        db = SessionLocal()

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
                            ai_analysis = resume_analysis
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

        

#----------------------#
# Display Resume Information
#----------------------#

if st.session_state.cleaned_text:

    st.subheader("Cleaned Resume Text.")
    st.text_area("Resume Content", st.session_state.cleaned_text, height=400)
    st.write(f"Candidate ID: {st.session_state.candidate_id}")
    st.write(f"Resume ID: {st.session_state.resume_id}")

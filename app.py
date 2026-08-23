import os
import shutil

import streamlit as st

from database.db import SessionLocal
from database.models import Candidate, Resume
from services.resume_parser import extract_resume_text
from services.text_cleaner import clean_resume_text
from services.llm import analyze_resume, match_resume_with_job

UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

st.set_page_config(page_title = "AI Resume Matcher",
                   page_icon = "📄",
                   layout = "wide"

)

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



if st.button("Upload Resume"):

    if not candidate_name:
        st.error("Please enter the candidate name.")
    elif uploaded_file is None:
            st.error("Please upload a resume file.")
    else:
        try:
                #----------------------#
                # 1. Save Uploaded File 
                #----------------------#

                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as file:
                    file.write(uploaded_file.getbuffer())


                #----------------------#
                # 2. Extract Resume Text
                #----------------------#

                resume_text = extract_resume_text(file_path)
                if not resume_text:
                    st.error("Could not extract text from the uploaded resume.")
                else:

                    #----------------------#
                    # 2.1 Clean Resume Text
                    #----------------------#

                    cleaned_text = clean_resume_text(resume_text)

                    # store resume data in session state

                    st.session_state.resume_text = resume_text
                    st.session_state.cleaned_text = cleaned_text

                    #----------------------#
                    # 3. Create Database Session
                    #----------------------#

                    db = SessionLocal()

                    #----------------------#
                    # 4. Create Candidate
                    #----------------------#

                    candidate = Candidate(name=candidate_name, email=candidate_email, phone=candidate_phone)

                    db.add(candidate)
                    db.commit()
                    db.refresh(candidate)

                    #----------------------#
                    # 5. Create Resume Record
                    #----------------------#

                    resume = Resume(candidate_id=candidate.id, filename= uploaded_file.name, file_path=file_path, 
                                    raw_text=cleaned_text)
                    db.add(resume)
                    db.commit()
                    db.refresh(resume)

                    #----------------------#
                    # 6. Store IDs
                    #----------------------#

                    candidate_id = candidate.id
                    resume_id = resume.id

                    # store IDs in session state

                    st.session_state.candidate_id = candidate_id
                    st.session_state.resume_id = resume_id

                    #----------------------#
                    # 7. Close Database
                    #----------------------#

                    db.close()

                    #----------------------#
                    # 8. Analyze Resume with LLM
                    #----------------------#

                    resume_analysis = analyze_resume(resume_text)

                    # store LLM analysis in session state

                    st.session_state.resume_analysis = resume_analysis

                    st.success("Resume uploaded and information stored successfully.")

        
        except Exception as e:
            
            st.error(f"An Error Occurred:{str(e)}")


# Display AI Resume Ananlysis outside Upload Button

if st.session_state.resume_analysis:
    
    st.subheader("AI Resume Analysis")

    st.write(st.session_state.resume_analysis)

#----------------------#
# 9. Job Description
#----------------------#

if st.session_state.cleaned_text:
    
    st.subheader("Job Description")

    job_description = st.text_area("Paste the job description here:", height = 200)


#----------------------#
# 10. Resume Job Matching
#----------------------#

if st.button("Match Resume with Job"):
    
    if not job_description.strip():

        st.warning("Please enter a job description.")

    else:

        try:

            #-------------------#
            # Match Resume with Job
            #-------------------#

            with st.spinner(
                "Matching resume with job..."
            ):
            
                match_result = match_resume_with_job(
                                st.session_state.cleaned_text,
                                job_description
                            )

                            #---------------#
                            # 10.1 Display Match Result
                            #---------------#

            if "error" in match_result:
                st.error(match_result["error"])
                st.write(match_result["raw_response"])

            else:
                st.subheader("AI Resume Match Result")
                st.metric("Match Score", f"{match_result['match_score']}%")
                st.write(
                    f"**Recommendation:**"
                    f"{match_result['recommendation']}"
                )
                st.write("###Matching Skills")

                for skill in match_result["matching_skills"]:
                    st.write(f"✓ {skill}")

                st.write("###Missing Skills")

                for skill in match_result["missing_skills"]:
                    st.write(f"✗ {skill}")

                st.write("### Experience Match")

                st.write(match_result["experience_match"])

                st.write("### Education Match")

                st.write(match_result["education_match"])

                st.write("### Strengths")

                for strength in match_result["strengths"]:
                    st.write(f"• {strength}")

                st.write("### Weaknesses")

                for weakness in match_result["weaknesses"]:
                    st.write(f"• {weakness}")

                st.write("### Explanation")

                st.write(match_result["explanation"])


        except Exception as e:

            st.error(f"Matching Error: {str(e)}")



#----------------------#
# 11. Display Resume Information
#----------------------#
        
st.subheader("Cleaned Resume Text.")
st.text_area("Resume Content", st.session_state.cleaned_text, height=400)
st.write(f"Candidate ID: {st.session_state.candidate_id}")
st.write(f"Resume ID: {st.session_state.resume_id}")

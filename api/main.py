from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
import os
import shutil

from database.db import SessionLocal
from database.models import Candidate, Resume
from services.resume_parser import extract_resume_text
from services.text_cleaner import clean_resume_text
from services.tfidf_matcher import calculate_tfidf_match
from services.embeddings import calculate_semantic_similarity
from services.llm import match_resume_with_job, analyze_resume_with_rag
from services.ml_matcher import predict_ml_match
from services.rag_service import retrieve_resume_context
from services.scoring_engine import calculate_final_score, get_recommendation


#------------------
# FastAPI
#------------------

app = FastAPI(
    title = "AI Resume Matcher API",
    description = "BAckend API for AI Resume Matcher",
    version = "1.0.0"
)


class MatchRequest(BaseModel):
    resume_id: int
    job_description: str


class MatchResponse(BaseModel):
    message: str
    resume_id: int

    tfidf_score: float
    semantic_score: float
    final_score: float
    recommendation: str

    ml_score: float

    rag_context: str
    rag_analysis: str

    match_result: dict


UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok = True)


#-------------------
# GET API
#-------------------


@app.get("/")
def root():
    return {
        "message": "AI Resume Matcher API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


#-------------------
# POST API
#-------------------


@app.post("/upload-resume")
async def upload_resume(
    candidate_name: str = Form(...),
    candidate_email: str = Form(...),
    candidate_phone: str = Form(...),
    file: UploadFile = File(...)
):
    
    if not file.filename:
        raise HTTPException(
            status_code = 400,
            detail = "No file selected"
        )


    allowed_extensions = [".pdf", ".docx"]

    file_extension = os.path.splitext(file.filename)[1].lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code = 400,
            detail = "Only PDF and DOCX files are allowed."
        )


    file_path = os.path.join(
        UPLOAD_DIR,
        file.filename
    )

    try:

        # Save uploaded file

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)


        # Extract resume text

        resume_text = extract_resume_text(file_path)

        if not resume_text.strip():
            raise HTTPException(
                status_code = 400,
                detail = "Could not extract text from resume"
            )


        # Clean resume

        cleaned_text = clean_resume_text(resume_text)


        # Database Session

        db = SessionLocal()

        try:
            

            # Create Candidate

            candidate = Candidate(
                name = candidate_name,
                email = candidate_email,
                phone = candidate_phone
            )

            db.add(candidate)
            db.commit()
            db.refresh(candidate)


            # Create Resume

            resume = Resume(
                candidate_id = candidate.id,
                filename = file.filename,
                file_path = file_path,
                raw_text = cleaned_text
            )

            db.add(resume)
            db.commit()
            db.refresh(resume)


            # Store IDs before closing database session

            candidate_id = candidate.id
            resume_id = resume.id

        finally:
            db.close()

        return {
            "message": "Resume uploaded successfully.",
            "candidate_id": candidate_id,
            "resume_id": resume_id,
            "filename": file.filename
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code = 500,
            detail = f"Resume upload fail: {str(e)}"
        )


@app.post("/match-resume", response_model = MatchResponse)
def match_resume(request: MatchRequest):

    db = SessionLocal()

    try:

        #------------------
        # Find Resume
        #------------------

        resume = db.query(Resume).filter(
            Resume.id == request.resume_id
        ).first()

        if not resume:
            raise HTTPException(
                status_code = 404,
                detail = "Resume not found"
            )


        #-------------------------
        # Check Job Description
        #-------------------------

        if not request.job_description.strip():
            raise HTTPException(
                status_code = 404,
                detail = "Job description cannot be empty"
            )

        #---------------------
        # Get resume text
        #---------------------

        resume_text = resume.raw_text

        if not resume_text:
            raise HTTPException(
                status_code = 404,
                detail = "Resume text is empty"
            )

        #---------------------------
        # Calculate TF-IDF match
        #---------------------------

        tfidf_score = calculate_tfidf_match(
            resume_text,
            request.job_description
        )

        #--------------------------------
        # Calculate Semantic Similarity
        #--------------------------------

        semantic_score = calculate_semantic_similarity(
            resume_text,
            request.job_description
        )


        #------------------------
        # RAG Retreival
        #------------------------

        rag_context = retrieve_resume_context(
            request.job_description,
            n_results = 1
        )


        #--------------------------
        # Groq RAG Analysis
        #--------------------------

        rag_analysis = analyze_resume_with_rag(
            resume_text,
            request.job_description,
            rag_context
        )


        #-------------------------
        # LLM Resume Matching
        #-------------------------

        match_result = match_resume_with_job(
            resume_text,
            request.job_description
        )

        if "error" in match_result:
            raise HTTPException(
                status_code = 500,
                detail = match_result["error"]
            )


        #------------------------
        # Extract ML Features
        #------------------------

        matching_skills = match_result.get("matching_skills", [])
        missing_skills = match_result.get("missing_skills", [])

        if not isinstance(matching_skills, list):
            matching_skills = []

        if not isinstance(missing_skills, list):
            missing_skills = []

        total_skills = len(matching_skills) + len(missing_skills)

        if total_skills > 0: 
            skills_match = (len(matching_skills) / total_skills) * 100
            missing_skills_ratio = (len(missing_skills) / total_skills) * 100 

        else:
            skills_match = 0
            missing_skills_ratio = 0


        #------------------------
        # Experience Match
        #------------------------

        experience_text = str(
            match_result.get("experience_match", "")
        ).lower()

        if any(word in experience_text for word in ["strong", "excellent", "match", "yes"]):
            experience_match = 100

        elif any(word in experience_text for word in ["partial", "somewhat", "moderate"]):
            experience_match = 50

        else:
            experience_match = 0


        #-------------------------
        # Education Match
        #-------------------------

        education_text = str(
            match_result.get("education_match", "")
        )

        if any(word in education_text for word in ["strong", "excellent", "match", "yes"]):
            education_match = 100
        
        elif any(word in education_text for word in ["partial", "somewhat", "moderate"]):
            education_match = 50

        else:
            education_match = 0


        #----------------------
        # ML Prediction
        #----------------------

        ml_score = predict_ml_match(
            float(tfidf_score),
            float(semantic_score),
            skills_match,
            experience_match,
            education_match,
            missing_skills_ratio
        )


        #----------------------
        # Final Candidate Score
        #----------------------

        final_score = calculate_final_score(
            float(tfidf_score),
            float(semantic_score)
        )

        recommendation = get_recommendation(final_score)


        return {
            "message": "Resume matched successfully",
            "resume_id": resume.id,
            "tfidf_score": float(tfidf_score),
            "semantic_score": float(semantic_score),
            "final_score": float(final_score),
            "recommendation": recommendation,
            "ml_score": float(ml_score),
            "rag_context": rag_context,
            "rag_analysis": rag_analysis,
            "match_result": match_result
        }

    
    except Exception as e:

        import traceback 
        traceback.print_exc()

        raise HTTPException(
            status_code = 500,
            detail = str(e)
        )

    finally:
        db.close()
import os
import json
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is not set in the .env file.")

# Create Groq Client
client = Groq(api_key=GROQ_API_KEY)

# LLM Model
MODEL_NAME = "openai/gpt-oss-120b"

def generate_llm_response(prompt: str) -> str:
    """
    Send a prompt to the Groq LLM and return its response.
    """

    try:
        response = client.chat.completions.create(
            model = MODEL_NAME,
            messages = [
                {"role": "user", "content": prompt}
            ],
            temperature = 0.2
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"LLM Error: {str(e)}"


def analyze_resume(resume_text: str) -> str:
    """
    Analyze a candidate resume using the LLM.
    """

    prompt = f"""
    You are an expert AI recruitment assistant.

    Analyze the following candidate resume and create a professional candidate profile.

    CANDIDATE RESUME
    --------------------
    {resume_text}
    --------------------

    Provide the analysis using exactly these sections:

    1. Candidate Summary
       - Give a short professional summary of the candidate.

    2. Technical Skills
       - List the technical skills explicitly found in the resume.

    3. Soft Skills
       - List soft skills explicitly supported by the resume.

    4. Education
       - Mention degrees, institutions, fields of study, and relevant
         education information available in the resume.

    5. Work Experience
       - Summarize the candidate's work experience, including job titles,
         companies, responsibilities, and achievements when available.

    6. Projects
       - Summarize important projects mentioned in the resume.

    7. Certifications
       - List certifications or professional training if available.

    8. Experience Level
       - Estimate the candidate's experience level such as:
         Entry Level, Junior, Mid-Level, or Senior.
       - Only make this assessment using evidence from the resume.

    9. Strengths
       - Identify the candidate's strongest professional qualities based
         only on the resume.

    10. Areas for Improvement
        - Identify skills, experience, or qualifications that appear weak
          or limited based only on the resume.

    IMPORTANT RULES:
    - Do not invent information.
    - Only use information available in the resume.
    - If information is not available, say "Not mentioned in resume."
    - Do not compare the candidate with any job description.
    - Do not provide a match score.
    - Do not provide matching skills.
    - Do not provide missing job-specific skills.
    - Keep the analysis clear, professional, structured, and concise.


    Keep the analysis clear, professional, structured, and concise.
    """

    return generate_llm_response(prompt)


def match_resume_with_job(resume_text: str, job_description: str) -> str:
    """
    Compare a resume with a job description using the LLM.
    """

    prompt = f"""
    You are an expert AI recruitment assistant.

    Your task is to evaluate how well a candidate matches a job.

    CANDIDATE RESUME

    --------------------
    {resume_text}
    

    JOB DESCRIPTION
    --------------------
    {job_description}

    Return ONLY valid JSON.

    Use exactly this structure:

    {{
        "match_score": 0,
        "recommendation": "",
        "matching_skills": [],
        "missing_skills": [],
        "experience_match": "",
        "education_match": "",
        "strengths": [],
        "weaknesses": [],
        "explanation": ""
    }}

    Rules:

    - match _score must be a number from 0 to 100
    - recommendation must be one of:
      Strong match
      Good match
      Partial match
      Weak match
    - Do not invent information
    - Only use information available in the resume and job description.
    """


    response = generate_llm_response(prompt)

    try:

        result = json.loads(response)

        return result

    except json.JSONDecodeError:

        return {
            "error": "Could not parse LLM response.",
            "raw_response": response
        }


def analyze_resume_with_rag(
    resume_text: str,
    job_description: str,
    rag_context: str
) -> str:
    
    prompt =  f"""
    You are an AI resume matching assistant.

    Analyze the candidate's resume against the given job description.

    Use the retrieved RAG context as supporting information.
    Do not invent skills, experience, education, or qualifications
    that are not present in the resume or retrieved context.

    Candidate Resume:
    {resume_text}

    Job Description:
    {job_description}

    Retrieved RAG Context:
    {rag_context}

    Provide a clear analysis containing:

    1. Overall Match
    2. Matching Skills
    3. Missing Skills
    4. Experience Match
    5. Education Match
    6. Strengths
    7. Weaknesses
    8. Final Recommendation
    9. Explanation

    Keep the analysis concise and relevant to the job description.
    """

    response = client.chat.completions.create(
        model = MODEL_NAME,
        messages = [
            {
                "role": "system",
                "content": "You are anexpert AI recruitment assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature = 0.2
    )

    return response.choices[0].message.content
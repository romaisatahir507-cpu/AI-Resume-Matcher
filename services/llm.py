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
    Analyze a candidate resumeusing the LLM.
    """

    prompt = f"""
    You are an expert AI recruitment assistant.

    Analyze the following candidate resume.

    Resume:
    --------------------
    {resume_text}
    --------------------

    Provide the following information:
    1. Candidate's main skills
    2. Technical skills
    3. Soft skills
    4. Education
    5. Work experience
    6. Years of experience if available
    7. Main areas of expertise
    8. Potential job roles
    9. Strengths
    10. Missing or weak areas

    Keep the analysis clear, structured, and concise.
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
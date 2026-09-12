import requests

API_URL = "http://127.0.0.1:8000"

def match_resume(resume_id: str, job_description: str):

    print("resume_id:", resume_id)
    print("job_description:", job_description)

    response = requests.post(
        f"{API_URL}/match-resume",
        json = {
            "resume_id": resume_id,
            "job_description": job_description
        }
    )

    print("status:", response.status_code)
    print("response:", response.text)

    response.raise_for_status()

    return response.json()
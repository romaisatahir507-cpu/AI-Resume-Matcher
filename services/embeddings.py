from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# Load embedding Model

model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_semantic_similarity(resume_text: str, job_description:str) -> float:
    """
    Calculate semantic similarity between resume and job description. 
    Return a percentage between 0 and 100.
    """

    if not resume_text.strip() or not job_description.strip():
        return 0.0

    
    # Convert both texts into embeddings
    resume_embedding = model.encode([resume_text])
    job_embedding = model.encode([job_description])

    # Calculate Cosine Similarity
    similarity = cosine_similarity(
        resume_embedding,
        job_embedding
    )[0][0]

    # Convert to percentage

    score = similarity * 100

    return round(score, 2)


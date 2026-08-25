from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf_match(resume_text: str, job_description: str ) -> float:
    """
    Calculate similarity between a resume and job description using TF-IDF and cosine similarity.
    """

    documents = [resume_text, job_description]

    vectorizer = TfidfVectorizer(stop_words = "english")

    tfidf_matrix = vectorizer.fit_transform(documents)

    similarity_score = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )[0][0]

    match_percentage = similarity_score * 100

    return round(match_percentage, 2)
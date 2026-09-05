import chromadb
from sentence_transformers import SentenceTransformer


# ChromaDB persistent storage
client = chromadb.PersistentClient(path="chroma_db")


# Create or get resume collection
collection = client.get_or_create_collection(name="resume_collection")


# Embedding Model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def add_resume_to_chroma(
    resume_id: int,
    resume_text: str
):
    """
    Store resume text and its embedding in ChromaDB.
    """

    embedding = embedding_model.encode(
        resume_text
    ).tolist()

    collection.upsert(
        ids=[str(resume_id)],
        documents=[resume_text],
        embeddings=[embedding],
        metadatas=[
            {
                "resume_id": resume_id
            }
        ]
    )



def search_resumes(query_text: str, n_results: int = 3):
    """
    Search ChromaDB for resumes similar to the query.
    """

    query_embedding = embedding_model.encode(query_text).tolist()

    results = collection.query(
        query_embeddings = [query_embedding],
        n_results = n_results
    )

    return results
from services.chroma_service import search_resumes

def retrieve_resume_context(
    job_description: str,
    n_results: int = 3
):

    """
    Retrieve the most relevant resume information from ChromaDB using the job
    description.
    """

    results = search_resumes(
        query_text=job_description,
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]

    if not documents:
        return ""


    context = "\n\n".join(documents)

    return context


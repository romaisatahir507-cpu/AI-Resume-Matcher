import sys
import os

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)


from collections import defaultdict
import hashlib
import os

from database.db import SessionLocal
from database.models import Candidate, Resume

# ChromaDB
from services.chroma_service import collection


#--------------------------------
# Generate Resume Hash
#--------------------------------

def generate_resume_hash(resume_text):

    if not resume_text:
        return None

    normalized_text = " ".join(
        resume_text.lower().split()
    )

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


#--------------------------------
# Start Database Session
#--------------------------------

db = SessionLocal()

try:

    resumes = (
        db.query(Resume)
        .order_by(Resume.id.asc())
        .all()
    )

    print("\n==============================")
    print("DUPLICATE RESUME CLEANUP")
    print("==============================\n")

    print(f"Total resumes before cleanup: {len(resumes)}")

    #--------------------------------
    # Group resumes by content
    #--------------------------------

    resume_groups = defaultdict(list)

    for resume in resumes:

        resume_hash = generate_resume_hash(
            resume.raw_text
        )

        resume_groups[resume_hash].append(resume)

    print(
        f"Unique resume contents: "
        f"{len(resume_groups)}"
    )

    #--------------------------------
    # Determine records to keep/delete
    #--------------------------------

    resumes_to_keep = []
    resumes_to_delete = []

    for resume_hash, group in resume_groups.items():

        # Oldest resume = first because ordered by ID
        keep_resume = group[0]

        resumes_to_keep.append(
            (keep_resume, resume_hash)
        )

        # Remaining records are duplicates
        for duplicate_resume in group[1:]:

            resumes_to_delete.append(
                duplicate_resume
            )

    print(
        f"Resumes to keep: "
        f"{len(resumes_to_keep)}"
    )

    print(
        f"Duplicate resumes to delete: "
        f"{len(resumes_to_delete)}"
    )

    print("\n------------------------------")
    print("RESUMES THAT WILL BE KEPT")
    print("------------------------------\n")

    for resume, resume_hash in resumes_to_keep:

        candidate = (
            db.query(Candidate)
            .filter(
                Candidate.id == resume.candidate_id
            )
            .first()
        )

        candidate_name = (
            candidate.name
            if candidate
            else "Unknown"
        )

        print(
            f"Resume ID: {resume.id} | "
            f"Candidate: {candidate_name} | "
            f"File: {resume.filename}"
        )

    print("\n------------------------------")
    print("RESUMES THAT WILL BE DELETED")
    print("------------------------------\n")

    for resume in resumes_to_delete:

        candidate = (
            db.query(Candidate)
            .filter(
                Candidate.id == resume.candidate_id
            )
            .first()
        )

        candidate_name = (
            candidate.name
            if candidate
            else "Unknown"
        )

        print(
            f"Resume ID: {resume.id} | "
            f"Candidate ID: {resume.candidate_id} | "
            f"Candidate: {candidate_name}"
        )

    #--------------------------------
    # Safety Confirmation
    #--------------------------------

    print("\n==============================")
    print("WARNING")
    print("==============================")

    print(
        "The duplicate Resume records, "
        "their duplicate Candidates, and "
        "their ChromaDB entries will be deleted."
    )

    confirmation = input(
        "\nType DELETE to continue: "
    )

    if confirmation != "DELETE":

        print("\nCleanup cancelled.")
        db.close()
        raise SystemExit

    #--------------------------------
    # Update hashes for kept resumes
    #--------------------------------

    for resume, resume_hash in resumes_to_keep:

        resume.resume_hash = resume_hash

    db.flush()

    #--------------------------------
    # Delete duplicate ChromaDB entries
    #--------------------------------

    print("\nRemoving duplicate ChromaDB entries...")

    for resume in resumes_to_delete:

        try:

            collection.delete(
                ids=[str(resume.id)]
            )

        except Exception as e:

            print(
                f"Could not delete ChromaDB entry "
                f"for Resume ID {resume.id}: {e}"
            )

    #--------------------------------
    # Delete duplicate Resume records
    #--------------------------------

    print("Removing duplicate Resume records...")

    duplicate_candidate_ids = set()

    for resume in resumes_to_delete:

        duplicate_candidate_ids.add(
            resume.candidate_id
        )

        db.delete(resume)

    db.flush()

    #--------------------------------
    # Delete duplicate Candidates
    #--------------------------------

    print("Removing duplicate Candidate records...")

    for candidate_id in duplicate_candidate_ids:

        candidate = (
            db.query(Candidate)
            .filter(
                Candidate.id == candidate_id
            )
            .first()
        )

        if candidate:

            # Only delete candidate if no resume remains
            remaining_resume = (
                db.query(Resume)
                .filter(
                    Resume.candidate_id == candidate_id
                )
                .first()
            )

            if not remaining_resume:

                db.delete(candidate)

    #--------------------------------
    # Commit Database Changes
    #--------------------------------

    db.commit()

    #--------------------------------
    # Final Report
    #--------------------------------

    remaining_resumes = (
        db.query(Resume).count()
    )

    remaining_candidates = (
        db.query(Candidate).count()
    )

    print("\n==============================")
    print("CLEANUP COMPLETED")
    print("==============================\n")

    print(
        f"Resumes remaining: "
        f"{remaining_resumes}"
    )

    print(
        f"Candidates remaining: "
        f"{remaining_candidates}"
    )

    print("\nDuplicate cleanup completed successfully.")


except Exception as e:

    db.rollback()

    print(
        f"\nCleanup failed. "
        f"Database changes were rolled back."
    )

    print(f"Error: {e}")


finally:

    db.close()
import sys
import os


# Add project root to Python path
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

from database.db import SessionLocal
from database.models import Candidate, Resume


db = SessionLocal()

try:

    orphan_candidates = (
        db.query(Candidate)
        .filter(~Candidate.resumes.any())
        .all()
    )

    print("==============================")
    print("ORPHAN CANDIDATE CLEANUP")
    print("==============================")

    print(f"\nOrphan candidates found: {len(orphan_candidates)}")

    for candidate in orphan_candidates:
        print(
            f"Candidate ID: {candidate.id} | "
            f"Name: {candidate.name}"
        )

    if not orphan_candidates:
        print("\nNo orphan candidates found.")
        db.close()
        sys.exit()

    print("\nThese candidates have NO resumes.")
    print("They will be permanently deleted.")

    confirmation = input("\nType DELETE to continue: ")

    if confirmation != "DELETE":
        print("\nCleanup cancelled.")
        db.close()
        sys.exit()

    for candidate in orphan_candidates:
        db.delete(candidate)

    db.commit()

    print("\n==============================")
    print("CLEANUP COMPLETED")
    print("==============================")

    remaining_candidates = db.query(Candidate).count()

    print(f"Candidates remaining: {remaining_candidates}")
    print(f"Resumes remaining: {db.query(Resume).count()}")

except Exception as e:

    db.rollback()
    print(f"\nError: {e}")

finally:
    db.close()
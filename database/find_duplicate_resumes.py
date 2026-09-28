import sqlite3
import hashlib
from collections import defaultdict

DB_PATH = "resume_matcher.db"


def generate_hash(text):

    if not text:
        return None

    normalized_text = " ".join(text.lower().split())

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT id, candidate_id, filename, raw_text
    FROM resumes
    ORDER BY id
""")

resumes = cursor.fetchall()

groups = defaultdict(list)

for resume_id, candidate_id, filename, raw_text in resumes:

    resume_hash = generate_hash(raw_text)

    if resume_hash:
        groups[resume_hash].append({
            "id": resume_id,
            "candidate_id": candidate_id,
            "filename": filename
        })


duplicate_groups = [
    group
    for group in groups.values()
    if len(group) > 1
]


print("\n==============================")
print("DUPLICATE RESUME REPORT")
print("==============================\n")

print(f"Total resume records: {len(resumes)}")
print(f"Unique resume contents: {len(groups)}")
print(f"Duplicate groups: {len(duplicate_groups)}")

duplicate_count = sum(
    len(group) - 1
    for group in duplicate_groups
)

print(f"Duplicate records: {duplicate_count}\n")


for number, group in enumerate(duplicate_groups, start=1):

    print(f"Duplicate Group {number}:")

    for resume in group:

        print(
            f"  Resume ID: {resume['id']} | "
            f"Candidate ID: {resume['candidate_id']} | "
            f"File: {resume['filename']}"
        )

    print()


conn.close()
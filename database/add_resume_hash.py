import sqlite3

DB_PATH = "resume_matcher.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

try:
    cursor.execute(
        "ALTER TABLE resumes ADD COLUMN resume_hash VARCHAR(64)"
    )

    conn.commit()

    print("resume_hash column added successfully.")

except sqlite3.OperationalError as e:

    if "duplicate column name" in str(e).lower():
        print("resume_hash column already exists.")
    else:
        print(f"Database error: {e}")

finally:
    conn.close()
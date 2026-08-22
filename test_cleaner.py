from services.text_cleaner import clean_resume_text

text = """
John Doe


Software Engineer


Python    Machine Learning    SQL
"""

cleaned = clean_resume_text(text)

print(cleaned)
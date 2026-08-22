import re

def clean_resume_text(text):
    if not text:
        return ""
    
    # Replace multiple spaces with one space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Remove leading/trailing spaces from lines
    lines = [line.strip() for line in text.splitlines()]

    # Remove empty lines
    lines = [line for line in lines if line]

    # Join lines
    cleaned_text = "\n".join(lines)

    return cleaned_text.strip()
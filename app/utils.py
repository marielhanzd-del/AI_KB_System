# app/utils.py
import os
from PyPDF2 import PdfReader
import docx

def extract_text_from_pdf(path):
    text = []
    reader = PdfReader(path)
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text.append(t)
    return "\n".join(text)

def extract_text_from_docx(path):
    doc = docx.Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def extract_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(path)
    if ext in [".docx", ".doc"]:
        return extract_text_from_docx(path)
    # fallback to plain text
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        return f.read()

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

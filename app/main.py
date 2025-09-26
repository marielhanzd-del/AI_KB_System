from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import google.generativeai as genai


app = FastAPI()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Create the model once and reuse it
MODEL = genai.GenerativeModel("gemini-1.5-flash")
# -------------------------------
# 1. Global variables
# -------------------------------
EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
DIM = EMBEDDING_MODEL.get_sentence_embedding_dimension()
INDEX = faiss.IndexFlatL2(DIM)
DOC_STORE = []  # keeps all text chunks in order
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -------------------------------
# 2. Helper: chunk text
# -------------------------------
def chunk_text(text, chunk_size=300, overlap=50):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# -------------------------------
# 3. Root route
# -------------------------------
@app.get("/")
def root():
    return {"message": "🚀 Knowledge Base API is running! Go to /docs to explore the endpoints."}

# -------------------------------
# 4. Upload documents
# -------------------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global INDEX, DOC_STORE

    try:
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Extract text (PDF only for now)
        text = ""
        if file.filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        else:
            text = "Non-PDF file uploaded."

        if not text.strip():
            return {"status": "uploaded", "warning": "No text extracted."}

        # Split into chunks
        chunks = chunk_text(text)

        # Encode chunks
        embeddings = EMBEDDING_MODEL.encode(
            chunks, convert_to_numpy=True
        ).astype("float32")

        # Add to FAISS + DOC_STORE
        INDEX.add(embeddings)
        DOC_STORE.extend(chunks)

        return {
            "status": f"Ingested {len(chunks)} chunks",
            "filename": file.filename,
            "total_chunks": len(DOC_STORE),
            "faiss_size": INDEX.ntotal,
            "text_sample": text[:200]
        }

    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# 5. Query documents
# -------------------------------
@app.post("/query")
async def query(question: str = Form(...)):
    global INDEX, DOC_STORE

    if INDEX.ntotal == 0:
        return {"answer": "No documents in the knowledge base. Please upload first."}

    # Encode query
    q_emb = EMBEDDING_MODEL.encode([question], convert_to_numpy=True).astype("float32")

    # Search FAISS
    k = 3
    D, I = INDEX.search(q_emb, k)
    results = [DOC_STORE[idx] for idx in I[0] if idx < len(DOC_STORE)]

    if not results:
        return {"answer": "I could not find an answer in the documents."}

    # Build context
    context = "\n\n".join(results)
    prompt = f"""
    You are a helpful assistant answering based only on the provided context.
    
    Question: {question}
    
    Context:
    {context}
    
    Give a clear and concise answer, grounded in the context.
    """

    try:
        resp = MODEL.generate_content(prompt)
        answer = resp.text.strip() if resp and resp.text else "No answer generated."
    except Exception as e:
        answer = f"Error generating answer: {str(e)}"

    return {
        "question": question,
        "answer": answer,
        "matched_chunks": results
    }
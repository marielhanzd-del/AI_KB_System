import os
from .vectorstore import FaissStore
from .db import SessionLocal, Chunk, Document
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import json, re

MODEL_NAME = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(MODEL_NAME)
store = FaissStore(dim=embed_model.get_sentence_embedding_dimension())

# Gemini API key (from environment)
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

def retrieve(query, top_k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = store.search(q_emb, top_k=top_k)
    db = SessionLocal()
    results = []
    for score_arr, idx_arr in zip(D, I):
        for score, idx in zip(score_arr, idx_arr):
            if idx == -1:
                continue
            chunk_id = store.id_map.get(int(idx))
            if chunk_id is None:
                continue
            ch = db.query(Chunk).filter(Chunk.id == chunk_id).first()
            doc = db.query(Document).filter(Document.id == ch.doc_id).first()
            results.append({
                "chunk_id": ch.id,
                "text": ch.text,
                "score": float(score),
                "doc": doc.filename
            })
    db.close()
    return results

def synthesize_answer(query, retrieved_chunks):
    context = "\n\n---\n\n".join([f"Source ({r['doc']}):\n{r['text']}" for r in retrieved_chunks])
    prompt = f"""Use the information below to answer the question. 
If the documents do not contain enough information, say you are uncertain and request missing information.

Context:
{context}

Question: {query}

Answer concisely and include a short list of sources used (filenames). 
After the answer, provide a one-sentence confidence level (low/medium/high).
"""
    if not GEMINI_KEY:
        return "LLM not configured. Here is the combined context:\n\n" + context[:2000]

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    return resp.text.strip()

def completeness_and_suggestions(query, answer, retrieved_chunks):
    prompt = f"""
Given the question and the provided context excerpts (sources are listed), determine if the answer is complete. 
If it's missing information, list the missing pieces and suggest concrete documents or actions to enrich the knowledge base 
(e.g., "upload product manual", "add API spec", "ingest database table X", or "add recent research paper 'title'"). 
Provide a confidence level (0-1).

Question: {query}
Answer: {answer}
Sources:
{chr(10).join([f"- {r['doc']}: {r['text'][:200]}" for r in retrieved_chunks])}

Return a JSON with keys: complete (true/false), confidence (0-1), missing_items (list of strings), suggestions (list of strings).
"""
    if not GEMINI_KEY:
        if len(retrieved_chunks) < 2:
            return {
                "complete": False,
                "confidence": 0.3,
                "missing_items": ["More sources needed"],
                "suggestions": ["upload manuals, add database table with events, add CSV of logs"]
            }
        return {"complete": True, "confidence": 0.8, "missing_items": [], "suggestions": []}

    model = genai.GenerativeModel("gemini-1.5-flash")
    resp = model.generate_content(prompt)
    raw = resp.text

    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            return {
                "complete": False,
                "confidence": 0.0,
                "missing_items": ["could not parse Gemini output"],
                "suggestions": []
            }
    return {
        "complete": False,
        "confidence": 0.0,
        "missing_items": ["no json in LLM"],
        "suggestions": []
    }

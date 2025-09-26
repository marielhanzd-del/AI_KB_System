# app/ingest.py
from sentence_transformers import SentenceTransformer
from .db import SessionLocal, Document, Chunk
from .vectorstore import FaissStore
from .utils import extract_text, chunk_text
import numpy as np
import os
import uuid


MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)
store = FaissStore(dim=model.get_sentence_embedding_dimension())

def ingest_file(file_path, filename):
    import logging
    logging.basicConfig(level=logging.INFO)

    db = SessionLocal()
    text = extract_text(file_path)
    logging.info(f"Extracted {len(text)} characters from {filename}")

    # save document (with unique filename)
    base, ext = os.path.splitext(filename)
    filename = f"{base}_{uuid.uuid4().hex[:8]}{ext}"

    doc = Document(filename=filename, text=text)
    db.add(doc)
    db.commit()
    db.refresh(doc)

    chunks = chunk_text(text)
    logging.info(f"Split into {len(chunks)} chunks")

    chunk_ids = []
    for c in chunks:
        ch = Chunk(doc_id=doc.id, text=c)
        db.add(ch)
        db.flush()
        chunk_ids.append(ch.id)

    db.commit()

    chunk_texts = [c.text for c in db.query(Chunk).filter(Chunk.doc_id == doc.id).all()]
    embeddings = model.encode(chunk_texts, convert_to_numpy=True, show_progress_bar=True)
    store.add_embeddings(embeddings.astype("float32"), chunk_ids)

    db.close()
    logging.info(f"Ingest complete: doc_id={doc.id}, chunks={len(chunk_texts)}")

    return {"doc_id": doc.id, "n_chunks": len(chunk_texts)}
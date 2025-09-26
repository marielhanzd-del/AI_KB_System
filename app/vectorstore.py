# app/vectorstore.py
import faiss
import numpy as np
import os
import pickle

INDEX_DIR = "./indexes"
os.makedirs(INDEX_DIR, exist_ok=True)
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
MAPPING_PATH = os.path.join(INDEX_DIR, "id_map.pkl")

class FaissStore:
    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.id_map = {}  # map of faiss_id -> chunk metadata
        self.next_id = 0

    def add(self, vectors, metadata):
        self.index.add(vectors)
        for m in metadata:
            self.id_map[self.next_id] = m
            self.next_id += 1
    def save(self):
        faiss.write_index(self.index, INDEX_PATH)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(self.id_map, f)

    def add_embeddings(self, embeddings, chunk_ids):
        # embeddings: numpy array shape (n, dim)
        # chunk_ids: list of int
        # normalize to cosine
        faiss.normalize_L2(embeddings)
        start = self.index.ntotal
        self.index.add(embeddings)
        for i, cid in enumerate(chunk_ids):
            self.id_map[start + i] = cid
        self.save()

    def search(self, q_emb, top_k=5):
        faiss.normalize_L2(q_emb)
        D, I = self.index.search(q_emb, top_k)
        return D, I

    def ntotal(self):
        return self.index.ntotal

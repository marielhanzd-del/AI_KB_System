# app/db.py
from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from datetime import datetime

engine = create_engine("sqlite:///./kb.sqlite", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    text = Column(Text)

    chunks = relationship("Chunk", back_populates="document")

class Chunk(Base):
    __tablename__ = "chunks"
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey("documents.id"))
    text = Column(Text)
    faiss_idx = Column(Integer, index=True)  # position in FAISS index
    document = relationship("Document", back_populates="chunks")

def init_db():
    Base.metadata.create_all(bind=engine)

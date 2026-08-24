from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from database.db import Base

class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(150), nullable=False)
    email = Column(String(150), nullable=True)
    phone = Column(String(50), nullable=True)
    created_at = Column(DateTime, default = datetime.utcnow)
    resumes = relationship("Resume", back_populates = "candidate", cascade = "all, delete")


class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    candidate_id = Column(Integer, ForeignKey("candidates.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    raw_text = Column(Text, nullable=False)

    ai_analysis = Column(Text, nullable=True)


    created_at = Column(DateTime, default = datetime.utcnow)
    candidate = relationship("Candidate", back_populates="resumes")


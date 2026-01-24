"""
SQLAlchemy Models for Face Recognition System

Defines the database schema for SQL Server.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, LargeBinary, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Face(Base):
    """
    Face embeddings table.
    
    Stores user face data including the embedding vector for recognition.
    
    Attributes:
        id: Auto-increment primary key
        user_id: Unique user identifier (e.g., EMP001)
        name_user: User's display name
        image: Original face image as binary data
        embedding: 512-dimensional face embedding vector as binary (2048 bytes for float32)
        created_at: Record creation timestamp
    """
    __tablename__ = "faces"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50), unique=True, nullable=False, index=True)
    name_user = Column(String(100), nullable=True)
    image = Column(LargeBinary, nullable=True)  # VARBINARY(MAX) in SQL Server
    embedding = Column(LargeBinary, nullable=False)  # VARBINARY(MAX) - 512 * 4 bytes = 2048 bytes
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Face(id={self.id}, user_id='{self.user_id}', name='{self.name_user}')>"

from sqlalchemy import Column, Integer, String, Text, Float
from app.db.base import Base


class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)

    content = Column(Text)
    embedding = Column(Text)  # JSON list
    importance = Column(Float, default=0.5)

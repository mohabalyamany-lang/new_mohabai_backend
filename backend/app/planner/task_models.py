from sqlalchemy import Column, Integer, String, Text
from app.db.base import Base


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, index=True)

    goal = Column(Text)
    plan_json = Column(Text)
    status = Column(String, default="active")
    current_step = Column(Integer, default=0)

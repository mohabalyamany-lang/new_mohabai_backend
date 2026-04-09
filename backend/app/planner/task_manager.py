import json
from sqlalchemy.orm import Session

from app.planner.task_models import Task
from app.planner.plan_generator import plan_generator


class TaskManager:

    async def start_task(self, db: Session, user_id: int, goal: str):

        plan = await plan_generator.create_plan(goal)

        task = Task(
            user_id=user_id,
            goal=goal,
            plan_json=json.dumps(plan),
        )

        db.add(task)
        db.commit()
        db.refresh(task)

        return task

    def get_active_task(self, db: Session, user_id: int):
        return db.query(Task).filter(
            Task.user_id == user_id,
            Task.status == "active"
        ).first()


task_manager = TaskManager()

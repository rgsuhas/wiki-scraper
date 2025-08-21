from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
from datetime import datetime

class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

@dataclass
class Task:
    title: str
    status: TaskStatus = TaskStatus.TODO
    assigned_to: Optional[str] = None
    
class TaskManager:
    def __init__(self):
        self.tasks: List[Task] = []
    
    def create_task(self, title: str) -> Task:
        task = Task(title)
        self.tasks.append(task)
        return task
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[Task]:
        return [t for t in self.tasks if t.status == status]

"""
任务执行器
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class TaskExecutor(ABC):
    """任务执行器基类"""
    
    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        pass
    
    @abstractmethod
    def can_handle(self, task_type: str) -> bool:
        """判断是否能处理该类型任务"""
        pass


class TaskManager:
    """任务管理器"""
    
    def __init__(self):
        self.executors = []
    
    def register_executor(self, executor: TaskExecutor):
        """注册任务执行器"""
        self.executors.append(executor)
    
    def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务"""
        task_type = task.get("type")
        for executor in self.executors:
            if executor.can_handle(task_type):
                return executor.execute(task)
        return {"status": "error", "message": f"未知任务类型: {task_type}"}


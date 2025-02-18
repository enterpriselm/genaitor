from typing import Any, Dict, List, Optional, Union
from .base import Task, TaskResult, AgentRole
from ..llm import LLMProvider

class Agent:
    def __init__(
        self,
        role: Union[str, AgentRole],
        tasks: List[Task],
        llm_provider: Optional[LLMProvider] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.role = role if isinstance(role, AgentRole) else AgentRole.CUSTOM
        self.tasks = tasks
        self.llm_provider = llm_provider
        self.config = config or {}
        self.conversation_history: List[Dict[str, Any]] = []
        self.task_history: List[Dict[str, Any]] = []

    def execute_task(self, task: Task, input_data: Any) -> TaskResult:
        """Execute a single task"""
        try:
            result = task.execute(input_data)
            if task.config.validation_required:
                is_valid = task.validate_result(result)
                if not is_valid:
                    raise ValueError("Task result validation failed")
            return result
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

    def process_request(self, request: Any) -> TaskResult:
        """Process a request through all tasks"""
        for task in self.tasks:
            result = task.execute(request)
            if not result.success:
                return result
                
            # Atualiza o histórico
            self.task_history.append({
                "task": task.__class__.__name__,
                "input": request,
                "output": result
            })
            
            return result  # Retorna o resultado do primeiro task por enquanto

    def _update_conversation_history(self, request: Any, result: TaskResult) -> None:
        """Update the conversation history"""
        self.conversation_history.append({
            "request": request,
            "response": result.content,
            "metadata": result.metadata
        })

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the agent's conversation history"""
        return self.conversation_history 
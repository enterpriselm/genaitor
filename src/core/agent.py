import logging  # Added logging
from typing import Any, Dict, List, Optional, Union
from .base import Task, TaskResult, AgentRole
from ..llm import LLMProvider

# Configure logging
logging.basicConfig(level=logging.INFO)

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
            logging.info(f"Executing task: {task.description}")
            result = task.execute(input_data)
            if task.config.validation_required:
                is_valid = task.validate_result(result)
                if not is_valid:
                    raise ValueError("Task result validation failed")
            return result
        except Exception as e:
            logging.error(f"Error executing task '{task.description}': {str(e)}")  # Log the error
            return TaskResult(success=False, content=None, error=f"Error executing task '{task.description}': {str(e)}")

    async def process_request(self, request: Any, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        """Process a request through all tasks, optionally using context"""
        if context:
            request = self._integrate_context(request, context)

        for task in self.tasks:
            result = self.execute_task(task, request)
            if not result.success:
                return result
                
            self.task_history.append({
                "task": task.__class__.__name__,
                "input": request,
                "output": result
            })
            
            return result  # Return the result of the first task for now

    def _integrate_context(self, request: Any, context: Dict[str, Any]) -> Any:
        """Integrate context into the request (custom logic can be added here)"""
        # Example logic to modify the request based on context
        if isinstance(request, dict) and isinstance(context, dict):
            # Merge context into the request dictionary
            request.update(context)
        elif isinstance(request, str):
            # If the request is a string, append context information
            context_info = " ".join(f"{key}: {value}" for key, value in context.items())
            request = f"{request} | Context: {context_info}"
        
        return request  # Return the modified request

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

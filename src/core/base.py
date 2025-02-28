from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

class AgentRole(Enum):
    MAIN = "main"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    REFINER = "refiner"
    SUMMARIZER = "summarizer"
    SCIENTIST = "scientist"
    ENGINEER = "engineer"
    ARCHITECT = "atchitect"
    CUSTOM = "custom"

@dataclass
class TaskConfig:
    """Configuration parameters for task execution"""
    max_retries: int = 3
    timeout: int = 60
    validation_required: bool = True
    cache_results: bool = True

@dataclass
class TaskResult:
    """Structured result from task execution"""
    success: bool
    content: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

class Task(ABC):
    def __init__(
        self,
        description: str,
        goal: str,
        output_format: str,
        config: Optional[TaskConfig] = None
    ):
        self.description = description
        self.goal = goal
        self.output_format = output_format
        self.config = config or TaskConfig()
        self.context: Dict[str, Any] = {}

    @abstractmethod
    def execute(self, input_data: Any) -> TaskResult:
        """Execute the task with given input"""
        pass

    def validate_result(self, result: TaskResult) -> bool:
        """Validate task execution result"""
        return True

    def update_context(self, data: Dict[str, Any]) -> None:
        """Update task context with new data"""
        self.context.update(data) 
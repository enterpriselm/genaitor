from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from enum import Enum
from google import genai
from PIL import Image

class AgentRole(Enum):
    MAIN = "main"
    SPECIALIST = "specialist"
    VALIDATOR = "validator"
    REFINER = "refiner"
    SUMMARIZER = "summarizer"
    SCIENTIST = "scientist"
    ENGINEER = "engineer"
    ARCHITECT = "architect"
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

class OCRImageAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.role = AgentRole.CUSTOM
        self.tasks = ["Extract text from image using Gemini API"]

    async def process_request(self, request: str, context: Optional[Dict[str, Any]] = None) -> TaskResult:
        try:
            image_path = request.strip()
            client = genai.Client(api_key=self.api_key)
            image = Image.open(image_path)
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[image, "Return only the readable texts extracted from the image, with no additional formatting or explanations."]
            )
            return TaskResult(success=True, content=response.text)
        except Exception as e:
            return TaskResult(success=False, content=None, error=f"OCR Error: {str(e)}")

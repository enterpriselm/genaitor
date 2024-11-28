from genaitor.request_helper import make_llama_request
from typing import Dict, Any, List
import re
import logging
from concurrent.futures import ThreadPoolExecutor
import os

logging.basicConfig(leven=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, role, system_message="", temperature=0.8, max_tokens=50, max_iterations=20):
        self.role = role
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
    
    def configure(self, **kwargs):
        """Dynamically update agent parameters."""
        allowed_attrs = {"temperature", "max_tokens", "max_iterations", "system_message"}
        for key, value in kwargs.items():
            if key in allowed_attrs:
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} does not exist in the agent.")

    def perform_task(self, prompt: str) -> Dict[str, Any]:
        sanitized_prompt = prompt.strip()
        logger.info(f"Performing task with prompt: {sanitized_prompt}")
        response = make_llama_request(
            sanitized_prompt,
            system_message=self.system_message, 
            temperature=self.temperature, 
            max_iterations=self.max_iterations, 
            max_tokens=self.max_tokens
        )
        if response.get("error"):
            logger.error(f"Error in {self.role}: {response['error']}")
            return {"error": f"{self.role} encountered an error: {response['error']}"}
        return response.get("content", "")


class Task:
    def __init__(self, description: str, expected_output: str, agent: Agent, goal: str = "", context: List[str] = None, output_file: str = None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context or []
        self.output_file = output_file
        self.goal = goal
        self.output = ""

    def execute(self, prompt: str) -> str:
        logger.info(f"Executing task: {self.description}")
        self.output = self.agent.perform_task(prompt)
        logger.debug(f"Task output: {self.output}")
        return self.output  


class Orchestrator:
    def __init__(self, agents: List[Agent], tasks: List[Task], process: str = 'sequential', cumulative: bool = False):
        self.agents = agents
        self.tasks = tasks
        self.process = process  # 'sequential' or 'parallel'
        self.cumulative = cumulative
        self.results = []

    def kickoff(self, **kwargs) -> Dict[str, Any]:
        """Executes the task flow."""
        if self.process == 'sequential':
            return self._sequential_execution(**kwargs)
        elif self.process == 'parallel':
            return self._parallel_execution(**kwargs)
        else:
            raise ValueError("Invalid process type. Choose 'sequential' or 'parallel'.")

    def _sequential_execution(self, **kwargs) -> Dict[str, Any]:
        """Execute tasks in sequence."""
        self.results.clear()
        prompt = ""
        for task in self.tasks:
            task.goal = self._replace_parameters(task.goal, kwargs)
            prompt += f"{task.agent.system_message} {task.description} Return: {task.expected_output}. {task.goal}".strip()
            result = task.execute(prompt)
            logger.info(f"Task result: {result}")
            self.results.append({task.description: result})
            prompt = result if self.cumulative else ""
        return {"output": self.results}

    def _parallel_execution(self, **kwargs) -> Dict[str, Any]:
        """Execute tasks in parallel."""
        self.results.clear()
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(task.execute, self._replace_parameters(task.goal, kwargs)): task
                for task in self.tasks
            }
            for future in futures:
                task = futures[future]
                try:
                    result = future.result()
                    logger.info(f"Task result for {task.description}: {result}")
                    self.results.append({task.description: result})
                except Exception as e:
                    logger.error(f"Error executing {task.description}: {e}")
                    self.results.append({task.description: {"error": str(e)}})
        return {"output": self.results}

    @staticmethod
    def _replace_parameters(goal: str, params: Dict[str, Any]) -> str:
        """Replace placeholders in task goals with actual values."""
        return re.sub(r"{(.*?)}", lambda m: params.get(m.group(1), ""), goal)
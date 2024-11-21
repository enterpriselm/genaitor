from utils.request_helper import make_llama_request
from multiprocessing import Pool, Manager
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class Agent:
    def __init__(self, role, goal="", system_message=""):
        self.role = role
        self.goal = goal
        self.system_message = system_message

    def perform_task(self, input_text):
        response = make_llama_request(input_text, system_message=self.system_message)
        if response.get("error"):
            return {"error": f"{self.role} encountered an error: {response['error']}"}
        return response["content"]

class Task(BaseModel):
    """Simplified Task class for execution.

    Attributes:
        description: A brief description of the task.
        expected_output: The desired result of the task.
        config: Optional configuration parameters for task execution.
        tools: Optional list of tools available for task execution.
    """
    description: str = Field(description="Description of the task.")
    expected_output: str = Field(description="Expected output for the task.")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration for the task.")
    tools: Optional[List[str]] = Field(default_factory=list, description="Tools available for this task.")
    output: Optional[Any] = Field(default=None, description="The result of task execution.")

    def execute(self) -> None:
        """Placeholder for executing the task.

        This should contain logic to perform the task using the description, tools, and config.
        """
        # Example placeholder logic
        self.output = f"Task '{self.description}' executed successfully."
        print(self.output)

    def __str__(self) -> str:
        return f"Task({self.id}): {self.description}"
        

class Orchestrator:
    def __init__(self, agents, tasks, process='sequential', cumulative=False):
        self.agents = agents
        self.tasks = tasks
        self.process = process  # 'sequential' or 'parallel'
        self.cumulative = cumulative
        self.results = []

    def kickoff(self, user_query):
        """Executes the task flow, passing output from one agent to the next."""
        # Initialize the cumulative input with the user's query
        if self.process == 'sequential':
            return self._sequential_execution(user_query)
        elif self.process == 'parallel':
            return self._parallel_execution(user_query)
        else:
            return {"error":"Invalid process type specified"}
        
    def _sequential_execution(self, user_query: str) -> Dict[str, Any]:
        """Executes tasks in a sequence, passing output from one to the next."""
        cumulative_input = user_query
        self.results.clear()

        for task in self.tasks:
            agent = self._get_agent_for_task(task)
            output = agent.perform_task(cumulative_input)
            if 'error' in output:
                return {"error": f"Error during task execution: {output['error']}"}

            self.results.append({task.description: output})
            cumulative_input = output

        return {"output": self.results}
    
    def _parallel_execution(self, user_query: str) -> Dict[str, Any]:
        """Executes tasks in parallel using multiprocessing, with each task receiving the initial user_query."""
        with Manager() as manager:
            results = manager.list()
            # Shared dictionary for cumulative input across tasks
            cumulative_data = manager.dict() if self.cumulative else None

            # Initialize cumulative_data with user query if cumulative is enabled
            if self.cumulative:
                cumulative_data['input'] = user_query

            # Prepare tasks for multiprocessing
            tasks_to_run = [
                (task, user_query, cumulative_data, results)
                for task in self.tasks
            ]

            # Use multiprocessing Pool to run tasks in parallel
            with Pool() as pool:
                pool.starmap(self._run_task, tasks_to_run)

            # Convert manager list to a standard list and return
            self.results = list(results)
            return {"output": self.results}

    def _run_task(self, task: Task, user_query: str, cumulative_data: Optional[Dict[str, str]], results: list):
        """Helper function to execute a task."""
        # Use cumulative input if enabled
        input_data = cumulative_data['input'] if cumulative_data else user_query

        # Execute the task
        task.execute(input_data)
        results.append({task.description: task.output})

        # Update cumulative data if cumulative interaction is enabled
        if cumulative_data:
            cumulative_data['input'] = task.output

    def _get_agent_for_task(self, task: Task) -> Agent:
        """Assigns an agent to a task based on task description"""
        if 'city selection' in task.description.lower():
            return self.agents['city_selection_agent']
        
        if 'local_guide' in task.description.lower():
            return self.agents['local_expert']
        
        if 'planner' in task.description.lower():
            return self.agents['travel_concierge']
        
        return self.agents.get('local_expert', None)
from genaitor.request_helper import make_llama_request
from typing import Dict, Any
import re


class Agent:
    def __init__(self, role, system_message="", temperature=0.8, max_tokens=50, max_iterations=20):
        self.role = role
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
    
    def configure(self, **kwargs):
        """Dynamically update agent parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise AttributeError(f"Attribute {key} does not exist in the agent.")

    def perform_task(self, prompt):
        response = make_llama_request(prompt, system_message=self.system_message, temperature=self.temperature, max_iterations=self.max_iterations, max_tokens=self.max_tokens)
        if response.get("error"):
            return {"error": f"{self.role} encountered an error: {response['error']}"}
        return response["content"]

class Task:
    def __init__(self, description, expected_output, agent, goal="", context=None, output_file=None):
        self.description = description
        self.expected_output = expected_output
        self.agent = agent
        self.context = context or []
        self.output_file = output_file
        self.goal = goal
        self.output = ""

    def execute(self, prompt):
        print(f"Agent: {self.agent.role}")
        print(f"Task: {self.description}")
        self.output = self.agent.perform_task(prompt)
        print(f"output: {self.output}")  # Debugging line
        return self.output    

class Orchestrator:
    def __init__(self, agents, tasks, process='sequential', cumulative=False):
        self.agents = agents
        self.tasks = tasks
        self.process = process  # 'sequential' or 'parallel'
        self.cumulative = cumulative
        self.results = []

    def kickoff(self, **kwargs):
        """Executes the task flow, passing output from one agent to the next."""
        # Initialize the cumulative input with the user's query
        if self.process == 'sequential':
            answer = self._sequential_execution(**kwargs)
        elif self.process == 'parallel':
            answer = self._parallel_execution(**kwargs)
        return {"output":answer}
    
    def _sequential_execution(self, **kwargs) -> Dict[str, Any]:
        """Executes tasks in a sequence, passing output from one to the next."""
        self.results.clear()
        prompt = ""
        for task in self.tasks:
            str_parameters = re.findall(r"{(.*?)}", task.goal)
            for parameter in str_parameters:
                task.goal = task.goal.replace("{"+parameter+"}", kwargs[parameter])
            prompt+=task.agent.system_message
            prompt+=task.description
            prompt+="You should return "
            prompt+=task.expected_output
            prompt+=task.goal
            prompt = prompt.replace("            ","") 
            result = task.execute(prompt)
            #print(f"Task result: {result}")
            self.results.append({task.description: result})
            if self.cumulative:
                prompt=result
            else:
                prompt=""
        return {"output": self.results}
    
    def _parallel_execution(self, user_query: str) -> Dict[str, Any]:
        pass

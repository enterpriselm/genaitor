from utils.request_helper import make_llama_request
from multiprocessing import Pool, Manager

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
        
    def _sequential_execution(self, user_query):
        """Executes tasks in a sequence, passing output from one to the next."""
        cumulative_input = user_query

        for task in self.tasks:
            output = task['agent'].perform_task(cumulative_input)
            if 'error' in output:
                return {"error": f"{task['agent'].role} task failed.", "details": output}

            self.results.append({task['description']: output})
            cumulative_input = output

        return {"output": self.results}
    
    def _parallel_execution(self, user_query):
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
                (task['agent'], task['description'], user_query, cumulative_data, results)
                for task in self.tasks
            ]

            # Use multiprocessing Pool to run tasks in parallel
            with Pool() as pool:
                pool.starmap(self._run_task, tasks_to_run)

            # Convert manager list to a standard list and return
            self.results = list(results)
            return {"output": self.results}

    def _run_task(self, agent, description, initial_input, cumulative_data, results):
        """Helper function to execute a task, optionally using and updating cumulative input."""
        # Use cumulative input if enabled; otherwise, use initial user query
        input_data = cumulative_data['input'] if cumulative_data else initial_input

        # Perform the task
        output = agent.perform_task(input_data)
        results.append({description: output})

        # Update cumulative data if cumulative interaction is enabled
        if cumulative_data:
            cumulative_data['input'] = output
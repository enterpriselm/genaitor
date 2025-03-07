import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

class DestinationTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Destination Selection", "Suggest a travel destination", "City, Country, and brief description")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        User Preferences: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "destination"})

class BudgetTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Budget Estimation", "Estimate travel costs", "Breakdown of expenses")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Destination and Preferences: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "budget"})

class ItineraryTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Itinerary Planning", "Create a travel schedule", "Day-wise activity list")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Destination and Budget: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "itinerary"})

destination_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[DestinationTask(llm_provider)],
    llm_provider=llm_provider
)

budget_agent = Agent(
    role=AgentRole.CUSTOM,
    tasks=[BudgetTask(llm_provider)],
    llm_provider=llm_provider
)

itinerary_agent = Agent(
    role=AgentRole.CUSTOM,
    tasks=[ItineraryTask(llm_provider)],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={
        "destination_agent": destination_agent,
        "budget_agent": budget_agent,
        "itinerary_agent": itinerary_agent
    },
    flows={
        "trip_planning_flow": Flow(
            agents=["destination_agent", "budget_agent", "itinerary_agent"],
            context_pass=[True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

result_process = orchestrator.process_request(
    "Looking for a tropical destination with a budget under $2000",
    flow_name="trip_planning_flow"
)
result = asyncio.run(result_process)

print(result)
import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

class TravelTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Input: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": self.description}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Travel Planning System...")
    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    provider = GeminiProvider(gemini_config)
    
    destination_selection_task = TravelTask(
        description="Destination Selection",
        goal="Suggest a travel destination",
        output_format="City, Country, and brief description",
        llm_provider=provider
    )
    
    budget_estimation_task = TravelTask(
        description="Budget Estimation",
        goal="Estimate travel costs",
        output_format="Breakdown of expenses",
        llm_provider=provider
    )
    
    itinerary_planning_task = TravelTask(
        description="Itinerary Planning",
        goal="Create a travel schedule",
        output_format="Day-wise activity list",
        llm_provider=provider
    )

    destination_selection_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[destination_selection_task],
        llm_provider=provider
    )
    budget_estimation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[budget_estimation_task],
        llm_provider=provider
    )
    
    itinerary_planning_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[itinerary_planning_task],
        llm_provider=provider
    )

    orchestrator = Orchestrator(
        agents={"destination_selection_agent": destination_selection_agent, 
                "budget_estimation_agent": budget_estimation_agent,
                "itinerary_planning_agent": itinerary_planning_agent},
        flows={
            "travel_planning_flow": Flow(agents=["destination_selection_agent", "budget_estimation_agent", "itinerary_planning_agent"], context_pass=[True, True, True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    travel_preferences = {
                            "budget": 2000,
                            "travel_period": "July 2025",
                            "preferred_continent": "Europe",
                            "climate_preference": "Warm",
                            "activity_preference": ["Beaches", "Cultural Experiences", "Nightlife"],
                            "travel_style": "Luxury",
                            "group_size": 2,
                            "food_preference": "Seafood",
                            "must_visit_places": ["Historical landmarks", "Local markets"]
                        }

    try:
        result = await orchestrator.process_request(travel_preferences, flow_name='travel_planning_flow')
        if result["success"]:
            with open('examples/files/travel_plan.txt', 'w') as f:
                f.write(result['content']['itinerary_planning_agent'].content.strip())
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
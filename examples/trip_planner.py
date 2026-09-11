import pandas as pd
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()
from pinnaitor.core import Orchestrator, Flow, ExecutionMode
from pinnaitor.presets.agents import create_preset_agents
from pinnaitor.presets.tasks import create_preset_tasks
from pinnaitor.presets.providers import create_gemini_provider

async def main(destination_selection_agent, budget_estimation_agent, itinerary_planning_agent):
    print("\nInitializing Travel Planning System...")
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
    provider = create_gemini_provider([os.getenv("GEMINI_API_KEY")])
    tasks = create_preset_tasks(provider)
    preset_agents = create_preset_agents(provider, tasks)
    asyncio.run(main(
        preset_agents["destination_selection_agent"],
        preset_agents["budget_estimation_agent"],
        preset_agents["itinerary_planning_agent"])) 
    

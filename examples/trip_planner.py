import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import destination_selection_agent, budget_estimation_agent, itinerary_planning_agent

async def main():
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
    asyncio.run(main())
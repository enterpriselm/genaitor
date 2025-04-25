import asyncio
import json

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import performance_agent, fatigue_agent, tactical_agent

async def main():
    print("\nInitializing Match Analysis System...")

    # Orquestração das tarefas
    orchestrator = Orchestrator(
        agents={
            "performance_agent": performance_agent,
            "fatigue_agent": fatigue_agent,
            "tactical_agent": tactical_agent
        },
        flows={
            "match_analysis_flow": Flow(
                agents=["performance_agent", "fatigue_agent", "tactical_agent"],
                context_pass=[True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    # Simulação de dados de entrada
    match_data = {
        "player_stats": {
            "passes_completed": 85,
            "distance_covered_km": 10.2,
            "duels_won": 12,
            "high_pressing_attempts": 18,
            "defensive_positioning": "medium block"
        },
        "fatigue_data": {
            "player_1": {"sprint_count": 45, "recovery_time": 2.1},
            "player_2": {"sprint_count": 20, "recovery_time": 3.8}
        },
        "tactical_data": {
            "possession_percentage": 60,
            "offensive_actions": 22,
            "defensive_errors": 3
        }
    }

    input_data = json.dumps(match_data, indent=4)

    print("\nStarting match analysis...")

    try:
        result = await orchestrator.process_request(input_data, flow_name='match_analysis_flow')

        if result["success"]:
            print("\nMatch Analysis Results:")
            print(result['content']['tactical_agent'].content)
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

import pandas as pd
import asyncio
from dotenv import load_dotenv
import os
load_dotenv()
from pinnaitor.core import Orchestrator, Flow, ExecutionMode
from pinnaitor.presets.agents import create_preset_agents
from pinnaitor.presets.tasks import create_preset_tasks
from pinnaitor.presets.providers import create_gemini_provider

async def main(feature_selection_agent, signal_analysis_agent, residual_evaluation_agent, lstm_model_agent, lstm_residual_evaluation_agent):
    print("\nInitializing Time Series Analysis System...")

    orchestrator = Orchestrator(
        agents={
            "feature_selection_agent": feature_selection_agent, 
            "signal_analysis_agent": signal_analysis_agent,
            "residual_evaluation_agent": residual_evaluation_agent,
            "lstm_model_agent": lstm_model_agent,
            "lstm_residual_evaluation_agent": lstm_residual_evaluation_agent
            # Adicionar Neural ODEs
        },
        flows={
            "time_series_analysis_flow": Flow(
                agents=["feature_selection_agent", "signal_analysis_agent", "residual_evaluation_agent", "lstm_model_agent", "lstm_residual_evaluation_agent"],
                context_pass=[True, True, True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    input_data = pd.read_csv(r'examples\files\temperature.csv')
    
    try:
        result = await orchestrator.process_request(input_data, flow_name='time_series_analysis_flow')
        i = 0
        if result["success"]:
            python_codes = result['content']['lstm_model_agent'].content.strip().split('```')
            for python_code in python_codes:
                if python_code.startswith('python'):
                    i += 1
                    filename = f'examples/files/time_series_analysis_{i}.py'
                    with open(filename, 'w') as f:
                        f.write(python_code.partition('python')[2])
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    provider = create_gemini_provider([os.getenv("GEMINI_API_KEY")])
    tasks = create_preset_tasks(provider)
    preset_agents = create_preset_agents(provider, tasks)
    asyncio.run(main(
        preset_agents["feature_selection_agent"],
        preset_agents["signal_analysis_agent"],
        preset_agents["residual_evaluation_agent"],
        preset_agents["lstm_model_agent"],
        preset_agents["lstm_residual_evaluation_agent"]
    )) 

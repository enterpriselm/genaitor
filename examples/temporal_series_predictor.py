import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import feature_selection_agent, signal_analysis_agent, residual_evaluation_agent, lstm_model_agent, lstm_residual_evaluation_agent

async def main():
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
    asyncio.run(main())

import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import data_understanding_agent, statistics_agent, anomalies_detection_agent, data_analysis_agent

async def main():
    print("\nInitializing Anomalies Detection System...")
    orchestrator = Orchestrator(
        agents={"data_understanding_agent": data_understanding_agent, 
                "statistics_agent": statistics_agent,
                "anomalies_detection_agent": anomalies_detection_agent,
                "data_analysis_agent": data_analysis_agent},
        flows={
            "anomalies_detection_flow": Flow(agents=["data_understanding_agent", "statistics_agent", "anomalies_detection_agent","data_analysis_agent"], context_pass=[True,True,True,True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    input_data = pd.read_csv(r'examples\files\temperature.csv')
    
    try:
        result = await orchestrator.process_request(input_data, flow_name='anomalies_detection_flow')
        i=0
        if result["success"]:
            python_codes = result['content']['data_analysis_agent'].content.strip().split('```')
            for python_code in python_codes:
                if python_code.startswith('python'):
                    i+=1
                    filename=f'examples/files/anomaly_detection_{i}.py'
                    with open(filename,'w') as f:
                        f.write(python_code.partition('python')[2])
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
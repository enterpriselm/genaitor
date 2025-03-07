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

class AnomaliesDetection(Task):
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
    print("\nInitializing Anomalies Detection System...")
    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    provider = GeminiProvider(gemini_config)
    
    data_understanding_task = AnomaliesDetection(
        description="Understanding Data",
        goal="Retrieve all information about a tabular dataset",
        output_format="Concise and informative",
        llm_provider=provider
    )
    
    statistics_task = AnomaliesDetection(
        description="Statistics Analysis",
        goal="Retrieve all the Statistics and general behavior of data on dataset",
        output_format="Bullet points or short paragraph",
        llm_provider=provider
    )
    
    anomalies_detection_task = AnomaliesDetection(
        description="Outliers Pattern Analysis",
        goal="Analyze and return the interval of data which seems to have anomalies compared with the data pattern",
        output_format="Bullet points or short paragraph",
        llm_provider=provider
    )

    data_analysis_task = AnomaliesDetection(
        description="Code Generation for Data Analysis",
        goal="Based on the previous analysis, generate a code in python to execute all of them.",
        output_format="Documentated, concize and complete python code",
        llm_provider=provider
    )

    data_understanding_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[data_understanding_task],
        llm_provider=provider
    )
    statistics_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[statistics_task],
        llm_provider=provider
    )

    anomalies_detection_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[anomalies_detection_task],
        llm_provider=provider
    )
    
    data_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[data_analysis_task],
        llm_provider=provider
    )

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
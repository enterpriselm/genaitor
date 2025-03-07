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

class TemporalSeriesForecasting(Task):
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
    print("\nInitializing Time Series Analysis System...")
    test_keys = [os.getenv('API_KEY')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    provider = GeminiProvider(gemini_config)
    
    feature_selection_task = TemporalSeriesForecasting(
        description="Feature Selection for Model Building",
        goal="Analyze dataset and determine which features are most relevant for modeling based on the target variable",
        output_format="List of features to use in the model",
        llm_provider=provider
    )

    signal_analysis_task = TemporalSeriesForecasting(
        description="Signal and Autoregressive Analysis",
        goal="Create models to analyze seasonality and trend in time series data",
        output_format="Python code to create AR or other models for time series analysis",
        llm_provider=provider
    )

    residual_evaluation_task = TemporalSeriesForecasting(
        description="Residual Evaluation for Signal Model",
        goal="Create code to evaluate the residuals of the signal model",
        output_format="Python code to evaluate residuals of the time series model",
        llm_provider=provider
    )

    lstm_model_task = TemporalSeriesForecasting(
        description="Build LSTM Model for Time Series Prediction",
        goal="Build a Long Short-Term Memory model for time series prediction based on the dataset",
        output_format="Python code for LSTM model with relevant hyperparameters",
        llm_provider=provider
    )

    lstm_residual_evaluation_task = TemporalSeriesForecasting(
        description="Residual Evaluation for LSTM Model",
        goal="Create code to evaluate the residuals of the LSTM model",
        output_format="Python code to evaluate residuals of the LSTM model",
        llm_provider=provider
    )

    neural_ode_task = TemporalSeriesForecasting(
        description="Build Neural ODE Model for Time Series Prediction",
        goal="Build a Neural ODE model for time series prediction based on the dataset",
        output_format="Python code for LSTM model with relevant hyperparameters",
        llm_provider=provider
    )

    neural_ode_evaluation_task = TemporalSeriesForecasting(
        description="Residual Evaluation for Neural ODE Model",
        goal="Create code to evaluate the residuals of the Neural ODE model",
        output_format="Python code to evaluate residuals of the Neural ODE model",
        llm_provider=provider
    )

    feature_selection_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[feature_selection_task],
        llm_provider=provider
    )
    
    signal_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[signal_analysis_task],
        llm_provider=provider
    )

    residual_evaluation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[residual_evaluation_task],
        llm_provider=provider
    )
    
    lstm_model_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[lstm_model_task],
        llm_provider=provider
    )
    
    lstm_residual_evaluation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[lstm_residual_evaluation_task],
        llm_provider=provider
    )

    orchestrator = Orchestrator(
        agents={
            "feature_selection_agent": feature_selection_agent, 
            "signal_analysis_agent": signal_analysis_agent,
            "residual_evaluation_agent": residual_evaluation_agent,
            "lstm_model_agent": lstm_model_agent,
            "lstm_residual_evaluation_agent": lstm_residual_evaluation_agent
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

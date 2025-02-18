import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, TaskResult,
    ExecutionMode, AgentRole
)
from src.llm import GeminiProvider, GeminiConfig

# Define custom task
class ForecastingTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Input Data: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "forecasting"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Forecasting Builder...")

    # Configurar Gemini com múltiplas chaves
    test_keys = [
        "AIzaSyCoC6voLEtOEOg5caWaqEIXBh8CiYWoUaY",
        "AIzaSyDA3r3LpI8cIGm4AVoaDQ65mDMD10GNTVM"
    ]
    
    # Configurar Gemini com limite de tokens
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000  # Limite de tokens por request
    )
    
    provider = GeminiProvider(gemini_config)
    
    # Criar agent
    forecasting_task = ForecastingTask(
        description="Generate forecasts based on input data.",
        goal="Provide accurate and helpful forecasts.",
        output_format="Clear, concise forecast",
        llm_provider=provider
    )
    
    agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[forecasting_task],
        llm_provider=provider
    )
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"gemini": agent},
        mode=ExecutionMode.SEQUENTIAL
    )
    
    # Test com dados de previsão
    input_data = [
        "Forecast sales for the next quarter based on previous trends.",
        "Predict the weather for the next week based on historical data.",
        "Estimate the growth of the tech industry in the next five years."
    ]
    
    # Process each input
    for data in input_data:
        print(f"\nInput Data: {data}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(data)
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("gemini")
                    if content and content.success:
                        print("\nResponse:")
                        print("-" * 80)
                        print(content.content.strip())
                    else:
                        print("Empty response received")
                else:
                    print(result["content"] or "Empty response")
            else:
                print(f"\nError: {result['error']}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    print("\nStarting Forecasting Builder...") 
    asyncio.run(main()) 
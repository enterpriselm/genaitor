import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, TaskResult, Flow,
    ExecutionMode, AgentRole
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

# Define custom task
class PinnTrainerTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Input Data: {input_data}
        
        Please provide a Python code snippet to train the PINN model based on the provided information.
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "pinn_trainer"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing PINN Trainer...")
    test_keys = [os.getenv('API_KEY_1'),os.getenv('API_KEY_2')]
    
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    pinn_trainer_task = PinnTrainerTask(
        description="Generate Python code to train a Physics-Informed Neural Network (PINN).",
        goal="Provide a clear and functional code snippet for training the PINN.",
        output_format="Python code for training the PINN",
        llm_provider=provider
    )
    
    agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[pinn_trainer_task],
        llm_provider=provider
    )
    
    orchestrator = Orchestrator(
        agents={"gemini": agent},
            flows={
            "default_flow": Flow(agents=["gemini"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    input_data_examples = [
        {
            "description": "Train a PINN for solving the 2D steady-state heat equation.",
            "geometry": "Square domain [0,1] x [0,1]",
            "boundary_conditions": "Left boundary: Temperature fixed at 100°C; Right boundary: Convective heat transfer with ambient temperature 20°C; Top and bottom boundaries: Insulated.",
            "equations": "∂²T/∂x² + ∂²T/∂y² = 0"
        },
        {
            "description": "Train a PINN to predict the population growth of a species using a logistic model.",
            "geometry": "Population growth in a closed environment.",
            "boundary_conditions": "Initial population: 100; Carrying capacity: 1000.",
            "equations": "dP/dt = rP(1 - P/K)"
        }
    ]
    
    # Process each input
    for example in input_data_examples:
        input_data = f"""
        Problem Description: {example['description']}
        Geometry: {example['geometry']}
        Boundary Conditions: {example['boundary_conditions']}
        Governing Equations: {example['equations']}
        """
        
        print(f"\nInput Data: {input_data.strip()}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(input_data, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("gemini")
                    if content and content.success:
                        print("\nTasks:")
                        print("-" * 80)
                        print("Python Code for Training the PINN:")
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
    print("\nStarting PINN Trainer...") 
    asyncio.run(main()) 
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
class PinnTask(Task):
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
                metadata={"task_type": "pinn"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing PINN Builder...")
    test_keys = [os.getenv('API_KEY_1'),os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    pinn_task = PinnTask(
        description="Build a Physics-Informed Neural Network (PINN) based on input data.",
        goal="Provide accurate and helpful PINN configurations.",
        output_format="Clear, concise PINN configuration",
        llm_provider=provider
    )
    
    agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[pinn_task],
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
            "description": "Solve the 2D steady-state heat equation in a square plate.",
            "geometry": "Square domain [0,1] x [0,1]",
            "boundary_conditions": "Left boundary: Temperature fixed at 100°C; Right boundary: Convective heat transfer with ambient temperature 20°C; Top and bottom boundaries: Insulated.",
            "equations": "∂²T/∂x² + ∂²T/∂y² = 0"
        },
        {
            "description": "Predict the population growth of a species using a logistic model.",
            "geometry": "Population growth in a closed environment.",
            "boundary_conditions": "Initial population: 100; Carrying capacity: 1000.",
            "equations": "dP/dt = rP(1 - P/K)"
        },
        {
            "description": "Model the diffusion of a pollutant in a river.",
            "geometry": "1D river flow.",
            "boundary_conditions": "Inlet concentration: 5 mg/L; Outlet concentration: 0 mg/L.",
            "equations": "∂C/∂t = D∂²C/∂x²"
        },
        {
            "description": "Analyze the motion of a pendulum.",
            "geometry": "Simple pendulum of length L.",
            "boundary_conditions": "Initial angle: θ0; No external forces.",
            "equations": "d²θ/dt² + (g/L)sin(θ) = 0"
        },
        {
            "description": "I want to model the airflow around an aircraft.",
            "geometry": "The aircraft has a wing shape with specific dimensions.",
            "boundary_conditions": "Inlet velocity: 50 m/s; Outlet pressure: atmospheric.",
            "equations": "Not specified."
        },
        {
            "description": "I would like to simulate heat transfer in a cylindrical rod.",
            "geometry": "Cylindrical rod of length 1m and radius 0.05m.",
            "boundary_conditions": "One end maintained at 100°C, the other end insulated.",
            "equations": "Not specified."
        },
        {
            "description": "I want to model a fluid flow problem without knowing the specific equations.",
            "geometry": "A channel with varying cross-section.",
            "boundary_conditions": "Inlet pressure: 100 kPa; Outlet pressure: 50 kPa.",
            "equations": "Not specified."
        }
    ]
    
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
                        print("PINN Configuration:")
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
    asyncio.run(main()) 
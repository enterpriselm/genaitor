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
    
# Define Base task
class GeneralTask(Task):
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
    print("\nInitializing Multi-Agent System...")
    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.2,
        verbose=False,
        max_tokens=2000
    )
    provider = GeminiProvider(gemini_config)
    
    # Define tasks
    equation_solver_task = GeneralTask(
        description="Solving Differential Equations",
        goal="Find analytical or numerical solutions to PDEs",
        output_format="Mathematical expressions or Python/NumPy code",
        llm_provider=provider
    )
    
    pinn_generation_task = GeneralTask(
        description="Generating Physics-Informed Neural Networks",
        goal="Create a neural network architecture tailored to solve PDEs",
        output_format="PyTorch model architecture",
        llm_provider=provider
    )
    
    hyperparameter_optimization_task = GeneralTask(
        description="Hyperparameter Tuning for PINNs",
        goal="Find optimal training parameters for PINN models",
        output_format="Dictionary of best hyperparameters",
        llm_provider=provider
    )

    orchestrator_task = GeneralTask(
        description="Orchestrating Adaptive Flow",
        goal="Determine which agent should handle the next part of the user input",
        output_format="Name of the next agent to handle the request",
        llm_provider=provider
    )

    validator_task = GeneralTask(
        description="Validating Adaptive Flow Response",
        goal="Determine if the agent's response is sufficient, or if another agent is needed",
        output_format="Decision ('complete' or the next agent's name)",
        llm_provider=provider
    )

    # Create agents
    equation_solver_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[equation_solver_task],
        llm_provider=provider
    )

    pinn_generation_agent = Agent(
        role=AgentRole.ENGINEER,
        tasks=[pinn_generation_task],
        llm_provider=provider
    )
    
    hyperparameter_optimization_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[hyperparameter_optimization_task],
        llm_provider=provider
    )

    orchestrator_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[orchestrator_task],
        llm_provider=provider
    )

    validator_agent = Agent(
        role=AgentRole.ENGINEER,
        tasks=[validator_task],
        llm_provider=provider
    )

    inputs = [
        "Solve the Schrödinger equation for a quantum harmonic oscillator.",
        "Design a PINN to approximate the Navier-Stokes equations in 2D.",
        "Optimize the learning rate and activation functions for a PINN solving wave equations."
    ]

    # Executing in Sequential Mode
    orchestrator_sequential = Orchestrator(
        agents={
            "solver_agent": equation_solver_agent,
            "pinn_agent": pinn_generation_agent,
            "optimizer_agent": hyperparameter_optimization_agent
        },
        flows={
            "sequential_flow": Flow(agents=["solver_agent", "pinn_agent", "optimizer_agent"], context_pass=[True, True, True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    print("\nExecuting Sequential Flow:")
    for input_data in inputs:
        result_process = orchestrator_sequential._process_sequential(input_data, flow_name='sequential_flow')
        result = asyncio.run(result_process)
        print(result)
    
    # Executing in Parallel Mode
    orchestrator_parallel = Orchestrator(
        agents={
            "solver_agent": equation_solver_agent,
            "pinn_agent": pinn_generation_agent,
            "optimizer_agent": hyperparameter_optimization_agent
        },
        flows={
            "parallel_flow": Flow(agents=["solver_agent", "pinn_agent", "optimizer_agent"], context_pass=[True, True, True])
        },
        mode=ExecutionMode.PARALLEL
    )
    print("\nExecuting Parallel Flow:")
    for input_data in inputs:
        result_process = orchestrator_parallel._process_parallel(input_data, flow_name='parallel_flow')
        result = asyncio.run(result_process)
        
        print(f"User Request: {input_data}")
        print("=="*20)
        print('')
        for agent in ["solver_agent", "pinn_agent", "optimizer_agent"]:
            print(f"Agent: {agent.capitalize()}\nAnswer: {result['content'][agent].content}")
            print("=="*20)

    # Executing in Adaptive Mode
    orchestrator_adaptive = Orchestrator(
        agents={
            "solver_agent": equation_solver_agent,
            "pinn_agent": pinn_generation_agent,
            "optimizer_agent": hyperparameter_optimization_agent,
            "orchestrator": orchestrator_agent,
            "validator": validator_agent
        },
        flows={
            "adaptive_flow": Flow(
                agents=["solver_agent", "pinn_agent", "optimizer_agent"], 
                context_pass=[True, True, True],
                orchestrator_agent="orchestrator",
                validator_agent="validator")
        },
        mode=ExecutionMode.ADAPTIVE
    )
    print("\nExecuting Adaptive Flow:")
    for input_data in inputs:
        result_process = orchestrator_adaptive._process_adaptative(input_data, flow_name='adaptive_flow')
        result = asyncio.run(result_process)
        
        print(f"User Request: {input_data}")
        print("=="*20)
        for agent in result['content'].keys():
            print(f"Agent: {agent.capitalize()}\nAnswer: {result['content'][agent].content}")
            print("=="*20)

if __name__ == "__main__":
    asyncio.run(main())

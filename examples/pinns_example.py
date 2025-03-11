import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import equation_solver_agent, pinn_generation_agent, hyperparameter_optimization_agent, orchestrator_agent, validator_agent

async def main():
    print("\nInitializing Multi-Agent System...")

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
    async def process_all(inputs):
        for input_data in inputs:
            result = await orchestrator_sequential._process_sequential(input_data, flow_name='sequential_flow')
            print(result)

    await process_all(inputs)
    
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
        result = await orchestrator_parallel._process_parallel(input_data, flow_name="parallel_flow")
        
        print(f"User Request: {input_data}")
        print("==" * 20)
        print("")
        
        for agent in ["solver_agent", "pinn_agent", "optimizer_agent"]:
            print(f"Agent: {agent.capitalize()}\nAnswer: {result['content'][agent].content}")
            print("==" * 20)

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
        result = await orchestrator_adaptive._process_adaptative(input_data, flow_name="adaptive_flow")

        print(f"User Request: {input_data}")
        print("==" * 20)

        for agent in result["content"]:
            print(f"Agent: {agent.capitalize()}\nAnswer: {result['content'][agent].content}")
            print("==" * 20)

if __name__ == "__main__":
    asyncio.run(main())
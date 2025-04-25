import os
import sys
import asyncio

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import problem_analysis_agent, numerical_analysis_agent, pinn_modeling_agent

async def main():
    print("\nInitializing FEM/FVM/FEA Problem Solver System...")
    orchestrator = Orchestrator(
        agents={"problem_analysis_agent": problem_analysis_agent,
                "numerical_modelling_agent": numerical_analysis_agent, 
                "pinn_modeling_agent": pinn_modeling_agent},
        flows={
            "problem_solving_flow": Flow(agents=["problem_analysis_agent", "numerical_modelling_agent", "pinn_modeling_agent"], context_pass=[True, True, True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    # Input data (for example: a simple heat conduction problem)
    user_requirements = "Solve a heat conduction problem in a 2D plate."
    problem_description = "Given the 2D heat conduction equation, choose the appropriate method (FEM/FVM/FEA), describe the methodology, and provide the solution with Python code."
    input_data = f"User Requirements:{user_requirements}\n\nProblem Description: {problem_description}"
    print("Starting problem-solving process to choose method and model PINN.\n")
    print(input_data)
    try:
        result = await orchestrator.process_request(input_data, flow_name='problem_solving_flow')
        if result["success"]:
            problem_analysis = result['content']['problem_analysis_agent'].content.strip()
            math_code = result['content']['numerical_modelling_agent'].content.strip()
            pinn_code = result['content']['pinn_modeling_agent'].content.strip()
            
            # Save Problem Analysis and PINN Code to Files
            with open('examples/files/problem_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(problem_analysis)
            
            with open('examples/files/numerical_modeling.py', 'w', encoding='utf-8') as f:
                f.write(math_code)

            with open('examples/files/pinn_modeling.py', 'w', encoding='utf-8') as f:
                f.write(pinn_code)
            
            print("Problem analysis:\n")
            print(problem_analysis)

            print("Numerical Modelling Code:\n")
            print(math_code)

            print("PINN Code:")
            print(pinn_code)
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

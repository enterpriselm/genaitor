import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
load_dotenv('.env')

class ProblemAnalysis(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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

class PINNModeling(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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
    print("\nInitializing FEM/FVM/FEA Problem Solver System...")
    test_keys = [os.getenv('API_KEY')]

    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=10000
    )
    provider = GeminiProvider(gemini_config)
    
    problem_analysis_task = ProblemAnalysis(
        description="Problem Analysis for FEM/FVM/FEA",
        goal="Analyze the problem and determine which approach (FEM, FVM, or FEA) is suitable for the given input.",
        output_format="Detailed analysis with methodology selection",
        llm_provider=provider
    )

    numerical_modeling_task = ProblemAnalysis(
        description="Solve the problem using the method recommended",
        goal="Solve the problem based on recommended method, using Python",
        output_format="Documented and full python code",
        llm_provider=provider
    )

    pinn_modeling_task = PINNModeling(
        description="PINN Modeling",
        goal="Model a Physics Informed Neural Network to solve the problem and compare it with FEM/FVM/FEA results.",
        output_format="Complete Python code to build and solve the PINN, with comparison of results",
        llm_provider=provider
    )
    
    problem_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[problem_analysis_task],
        llm_provider=provider
    )
    
    numerical_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[numerical_modeling_task],
        llm_provider=provider
    )
    
    pinn_modeling_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[pinn_modeling_task],
        llm_provider=provider
    )
    
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
    input_data = f"User Requirements:{user_requirements}/n/nProblem Description: {problem_description}"
    print("Starting problem-solving process to choose method and model PINN.")
    
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
            
            print("Problem analysis and PINN code generated successfully.")
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

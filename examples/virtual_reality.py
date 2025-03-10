import requests
from bs4 import BeautifulSoup
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

class UnityCodeGenerator(Task):
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
            return TaskResult(success=False, content=None, error=str(e))

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=5000
)
provider = GeminiProvider(gemini_config)

# Task para analisar requisitos e definir tecnologias
requirements_analysis = UnityCodeGenerator(
    description="Analyze Requirements",
    goal="Extract requirements for VR/AR Unity project",
    output_format="List of necessary Unity packages, SDKs, and scene components",
    llm_provider=provider
)

# Task para definir arquitetura do código
architecture_planning = UnityCodeGenerator(
    description="Plan Architecture",
    goal="Define the Unity scene structure and C# script organization",
    output_format="Detailed step-by-step guide on how the code should be structured",
    llm_provider=provider
)

# Task para gerar código C# baseado na arquitetura planejada
code_generation = UnityCodeGenerator(
    description="Generate Unity C# Code",
    goal="Generate complete and well-documented Unity C# scripts",
    output_format="Complete Unity C# script",
    llm_provider=provider
)

requirements_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[requirements_analysis],
    llm_provider=provider
)

architecture_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[architecture_planning],
    llm_provider=provider
)

code_generation_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[code_generation],
    llm_provider=provider
)

orchestrator = Orchestrator(
agents={
    "requirements_agent": requirements_agent,
    "architecture_agent": architecture_agent,
    "code_generation_agent": code_generation_agent
},
flows={
    "unity_vr_ar_flow": Flow(
        agents=["requirements_agent", "architecture_agent", "code_generation_agent"], 
        context_pass=[True, True, True]
    )
},
mode=ExecutionMode.SEQUENTIAL
)

async def main():

    print("\nStarting Unity VR/AR Code Generation System...")
    
    user_requirements = "Create a VR interaction system in Unity where users can pick up and move objects using Oculus controllers."
    
    input_data = f"User Requirements:\n{user_requirements}"
    
    try:
        result = await orchestrator.process_request(input_data, flow_name='unity_vr_ar_flow')
        
        if result["success"]:
            unity_code = result['content']['code_generation_agent'].content.strip()
            
            # Salvar código gerado
            filename = 'examples/files/generated_unity_code.cs'
            with open(filename, 'w') as f:
                f.write(unity_code)

            print(f"Unity code saved in {filename}")

        else:
            print(f"\nError: {result['error']}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
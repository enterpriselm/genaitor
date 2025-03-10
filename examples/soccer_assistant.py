import os
import asyncio
import sys
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Adicionar o caminho do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Carregar variáveis de ambiente
load_dotenv('.env')

class MatchAnalysisTask(Task):
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
    print("\nInitializing Match Analysis System...")

    test_keys = [os.getenv('API_KEY')]
    
    # Configuração do Gemini LLM
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=5000
    )
    provider = GeminiProvider(gemini_config)

    # Definição das tarefas
    performance_analysis = MatchAnalysisTask(
        description="Performance Analysis",
        goal="Analyze player performance based on real-time stats",
        output_format="JSON format with performance insights and improvement suggestions",
        llm_provider=provider
    )

    fatigue_detection = MatchAnalysisTask(
        description="Fatigue Detection",
        goal="Detect player fatigue and suggest adjustments",
        output_format="JSON format with player fatigue levels and recommended actions",
        llm_provider=provider
    )

    tactical_adjustment = MatchAnalysisTask(
        description="Tactical Adjustment",
        goal="Optimize team tactics based on match data",
        output_format="JSON format with suggested tactical changes",
        llm_provider=provider
    )

    # Criação dos agentes
    performance_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[performance_analysis],
        llm_provider=provider
    )
    
    fatigue_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[fatigue_detection],
        llm_provider=provider
    )

    tactical_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[tactical_adjustment],
        llm_provider=provider
    )

    # Orquestração das tarefas
    orchestrator = Orchestrator(
        agents={
            "performance_agent": performance_agent,
            "fatigue_agent": fatigue_agent,
            "tactical_agent": tactical_agent
        },
        flows={
            "match_analysis_flow": Flow(
                agents=["performance_agent", "fatigue_agent", "tactical_agent"],
                context_pass=[True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    # Simulação de dados de entrada
    match_data = {
        "player_stats": {
            "passes_completed": 85,
            "distance_covered_km": 10.2,
            "duels_won": 12,
            "high_pressing_attempts": 18,
            "defensive_positioning": "medium block"
        },
        "fatigue_data": {
            "player_1": {"sprint_count": 45, "recovery_time": 2.1},
            "player_2": {"sprint_count": 20, "recovery_time": 3.8}
        },
        "tactical_data": {
            "possession_percentage": 60,
            "offensive_actions": 22,
            "defensive_errors": 3
        }
    }

    input_data = json.dumps(match_data, indent=4)

    print("\nStarting match analysis...")

    try:
        result = await orchestrator.process_request(input_data, flow_name='match_analysis_flow')

        if result["success"]:
            print("\nMatch Analysis Results:")
            print(result['content']['tactical_agent'].content)
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

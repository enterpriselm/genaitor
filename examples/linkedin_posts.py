import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any

# Adiciona o caminho do projeto ao sistema
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Carrega variáveis de ambiente
load_dotenv('.env')

class SummarizationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: Dict[str, Any]) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input:
        {json.dumps(input_data, indent=4)}

        Provide the response in the following format:
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
    print("\nInicializando sistema de geração de posts para LinkedIn...")

    test_keys = [os.getenv('API_KEY')]

    # Configuração do LLM
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=5000
    )
    provider = GeminiProvider(gemini_config)

    # Tarefas definidas
    paper_summarization = SummarizationTask(
        description="Scientific Paper Summarization",
        goal="Summarize key points of a scientific paper",
        output_format="Bullet points highlighting main findings",
        llm_provider=provider
    )

    linkedin_post_generation = SummarizationTask(
        description="LinkedIn Post Generation",
        goal="Create an engaging LinkedIn post based on a scientific paper summary",
        output_format="A LinkedIn-friendly post with hashtags and a call to action",
        llm_provider=provider
    )

    # Agentes responsáveis
    summarization_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[paper_summarization],
        llm_provider=provider
    )

    linkedin_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[linkedin_post_generation],
        llm_provider=provider
    )

    # Orquestração do fluxo
    orchestrator = Orchestrator(
        agents={
            "summarization_agent": summarization_agent,
            "linkedin_agent": linkedin_agent
        },
        flows={
            "scientific_post_flow": Flow(
                agents=["summarization_agent", "linkedin_agent"],
                context_pass=[True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    # Teste com um artigo científico
    scientific_paper = """
    Este estudo apresenta um novo método de otimização baseado em redes neurais profundas para modelagem de fluidos.
    Os resultados indicam uma redução de 30% no erro de previsão em comparação com métodos tradicionais.
    Aplicações potenciais incluem engenharia aeroespacial e modelagem climática.
    """

    print("\nResumindo artigo científico...")

    try:
        result = await orchestrator.process_request(
            {"scientific_paper": scientific_paper},
            flow_name='scientific_post_flow'
        )

        if result["success"]:
            print("\nPost para LinkedIn gerado:")
            print(result['content'])
        else:
            print(f"\nErro: {result['error']}")

    except Exception as e:
        print(f"\nErro: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

load_dotenv('.env')

class EmailTask(Task):
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
    print("\n🚀 Inicializando sistema de geração de e-mails...")

    test_keys = [os.getenv('API_KEY')]

    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=5000
    )
    provider = GeminiProvider(gemini_config)

    # Definição das tarefas
    audience_research = EmailTask(
        description="Audience Research",
        goal="Analyze target audience and suggest the best email tone and approach",
        output_format="JSON format with audience insights and tone suggestions",
        llm_provider=provider
    )

    email_generation = EmailTask(
        description="Email Content Generation",
        goal="Generate an email draft based on the campaign and audience",
        output_format="Structured email content including subject line, body, and CTA",
        llm_provider=provider
    )

    email_optimization = EmailTask(
        description="Email Optimization",
        goal="Refine email content for clarity, engagement, and conversion",
        output_format="Optimized email content with persuasive elements",
        llm_provider=provider
    )

    email_personalization = EmailTask(
        description="Email Personalization",
        goal="Adapt email content for different audience segments",
        output_format="Different versions of the email for specific audience segments",
        llm_provider=provider
    )

    # Definição dos agentes
    research_agent = Agent(
        role=AgentRole.ANALYST,
        tasks=[audience_research],
        llm_provider=provider
    )

    content_agent = Agent(
        role=AgentRole.CREATOR,
        tasks=[email_generation],
        llm_provider=provider
    )

    optimization_agent = Agent(
        role=AgentRole.EDITOR,
        tasks=[email_optimization],
        llm_provider=provider
    )

    personalization_agent = Agent(
        role=AgentRole.MARKETER,
        tasks=[email_personalization],
        llm_provider=provider
    )

    # Orquestração do fluxo de e-mails
    orchestrator = Orchestrator(
        agents={
            "research_agent": research_agent,
            "content_agent": content_agent,
            "optimization_agent": optimization_agent,
            "personalization_agent": personalization_agent
        },
        flows={
            "email_marketing_flow": Flow(
                agents=["research_agent", "content_agent", "optimization_agent", "personalization_agent"],
                context_pass=[True, True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    # Exemplo de campanha de teste
    campaign_details = {
        "product": "Novo curso de Machine Learning",
        "audience": "Profissionais de tecnologia interessados em IA",
        "goal": "Gerar leads e aumentar conversões"
    }

    print("\n🔍 Analisando público-alvo...")

    try:
        result = await orchestrator.process_request(
            {"campaign_details": campaign_details},
            flow_name='email_marketing_flow'
        )

        if result["success"]:
            print("\n✉️ E-mail final gerado:")
            print(result['content'])
        else:
            print(f"\n❌ Erro: {result['error']}")

    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

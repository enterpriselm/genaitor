import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import research_agent, content_agent, optimization_agent, personalization_agent

async def main():
    print("\n🚀 Inicializando sistema de geração de e-mails...")

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

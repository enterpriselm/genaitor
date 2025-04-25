import asyncio
from dotenv import load_dotenv

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import summarization_agent, linkedin_agent

async def main():
    print("\nInicializando sistema de geração de posts para LinkedIn...")
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

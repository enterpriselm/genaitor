import sys
import asyncio
from dotenv import load_dotenv

from genaitor.core import Orchestrator, Flow, ExecutionMode
from genaitor.presets.agents import create_preset_agents
from genaitor.presets.tasks import create_preset_tasks
from genaitor.presets.providers import create_gemini_provider
from dotenv import load_dotenv
import os
load_dotenv()

async def main(research_agent, content_agent, optimization_agent, personalization_agent):
    print("\n🚀 Initializing generating e-mail systems...")

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

    campaign_details = {
        "product": "New Machine Learning course",
        "audience": "Tech professionals with interest in AI",
        "goal": "Generate leads and increase conversions"
    }

    print("\n🔍 Analyzing target public...")

    try:
        result = await orchestrator.process_request(
            {"campaign_details": campaign_details},
            flow_name='email_marketing_flow'
        )

        if result["success"]:
            print(result['content'])
        else:
            print(f"\n❌ Error: {result['error']}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    provider = create_gemini_provider([os.getenv("GEMINI_API_KEY")])
    tasks = create_preset_tasks(provider)
    preset_agents = create_preset_agents(provider, tasks)
    asyncio.run(main(
        preset_agents["research_agent"],
        preset_agents["content_agent"],
        preset_agents["optimization_agent"],
        preset_agents["personalization_agent"]
    )) 

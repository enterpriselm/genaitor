import os
import sys
import asyncio
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import research_agent, content_agent, optimization_agent, personalization_agent

async def main():
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
    asyncio.run(main())

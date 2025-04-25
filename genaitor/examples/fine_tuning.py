import asyncio

from llm.fine_tuning import FineTuningTask, ModelDeploymentTask
from core import Agent, AgentRole, Orchestrator, Flow, ExecutionMode

from presets.providers import gemini_provider
provider = gemini_provider()

async def main():
    model_name = "distilgpt2"
    dataset_name = "wikitext"
    output_dir = "./trained_model"
    
    fine_tuning_task = FineTuningTask(model_name, dataset_name, output_dir, provider)
    deploy_task = ModelDeploymentTask(output_dir, provider)

    fine_tuning_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[fine_tuning_task]
    )
    
    deploy_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[deploy_task]
    )
    
    orchestrator = Orchestrator(
        agents={"fine_tuning": fine_tuning_agent, "deploy": deploy_agent},
        flows={
            "fine_tune_and_deploy": Flow(agents=["fine_tuning", "deploy"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    result = await orchestrator.process_request(None, flow_name="fine_tune_and_deploy")
    
    if result["success"]:
        print("\nDeployment successful! API running at:")
        print(result["content"])
    else:
        print(f"\nError: {result['error']}")

if __name__ == "__main__":
    asyncio.run(main())
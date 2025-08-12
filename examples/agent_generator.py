import asyncio
import os

from genaitor.core import (
    Orchestrator, Flow, ExecutionMode
)
from genaitor.presets.agents import create_preset_agents
from genaitor.presets.providers import create_gemini_provider
from dotenv import load_dotenv
import os
load_dotenv()

async def main(agent_creation):
    print("\nInitializing Agent Creation Demo...")
    orchestrator = Orchestrator(
        agents={"creator": agent_creation},
        flows={
            "default_flow": Flow(agents=["creator"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Create an agent that can answer scientific questions related to physics.",
        "Design a task where an agent helps generate new PINN model architectures.",
        "Create an agent that helps with reviewing technical papers on machine learning."
    ]
    
    # Process each input and create a new agent
    for user_input in user_inputs:
        print(f"\nUser Input:\n\n {user_input}\n")
        print("=" * 80)
        print("\n")
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("creator")
                    if content and content.success:
                        print("\nGenerated Task for New Agent:\n")
                        print("-" * 80)
                        print("\n")
                             
                        formatted_text = content.content.strip()
                        
                        formatted_text = formatted_text.replace("**", "")
                        
                        for line in formatted_text.split('\n'):
                            if line.strip():
                                print(line)
                            else:
                                print()
                    else:
                        print("Empty response received")
                else:
                    print(result["content"] or "Empty response")
            else:
                print(f"\nError: {result['error']}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    provider = create_gemini_provider(["GEMINII API KEY"])
    preset_agents = create_preset_agents(provider)
    asyncio.run(main(preset_agents["agent_creation"]))

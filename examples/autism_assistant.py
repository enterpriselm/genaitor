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

async def main(autism_agent):
    print("\nInitializing Autism Assistant...")
    
    orchestrator = Orchestrator(
        agents={"gemini": autism_agent},
        flows={
            "default_flow": Flow(agents=["gemini"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    hyperfocus = 'Soccer'
    question = 'What is Data Science?'
    input_data = f"Hyperfocus: {hyperfocus}/nQuestion: {question}"
    print(input_data.replace('/n','\n'))
    print('\n')
    try:
        result = await orchestrator.process_request(input_data, flow_name='default_flow')
        if result["success"]:
            if isinstance(result["content"], dict):
                content = result["content"].get("gemini")
                if content and content.success:
                    print("\nResponse:\n")
                    print("-" * 80)
                    print('\n')
                    print(content.content.strip())
                else:
                    print("Empty response received")
            else:
                print(result["content"] or "Empty response")
        else:
            print(f"\nError: {result['error']}")
            
    except Exception as e:
        print(f"\nError: {str(e)}")
        
if __name__ == "__main__":
    provider = create_gemini_provider(["GEMINII API KEY"])
    preset_agents = create_preset_agents(provider)
    asyncio.run(main(preset_agents["autism_agent"])) 


import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import qa_agent, summarization_agent

async def main():
    print("\nInitializing Multi-Agent System...")
    orchestrator = Orchestrator(
        agents={"qa_agent": qa_agent, "summarization_agent": summarization_agent},
        flows={
            "default_flow": Flow(agents=["qa_agent", "summarization_agent"], context_pass=[True,True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    inputs = [
        "What is the impact of AI on modern healthcare?",
        "Summarize the latest research on quantum computing."
    ]
    
    for input_data in inputs:
        print(f"\nProcessing: {input_data}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(input_data, flow_name='default_flow')
            
            if result["success"]:
                for agent_name, content in result["content"].items():
                    if content and content.success:
                        print(f"\nResponse from {agent_name}:")
                        print("-" * 80)
                        print(content.content.strip())
                    else:
                        print(f"No valid response from {agent_name}")
            else:
                print(f"\nError: {result['error']}")
        
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    asyncio.run(main())
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import autism_agent

async def main():
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
    try:
        result = await orchestrator.process_request(input_data, flow_name='default_flow')
        if result["success"]:
            if isinstance(result["content"], dict):
                content = result["content"].get("gemini")
                if content and content.success:
                    print("\nResponse:")
                    print("-" * 80)
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
    asyncio.run(main())
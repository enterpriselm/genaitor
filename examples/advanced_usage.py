import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import qa_agent
    
async def main():
    print("\nInitializing Advanced Usage Demo...")
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"gemini": qa_agent},
        flows={
            "default_flow": Flow(agents=["gemini"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    questions = [
        "What is the difference between Python lists and tuples?",
        """Explain in detail how neural networks work, including: 
        1. Basic structure
        2. Different types of layers
        3. Activation functions
        4. Backpropagation
        5. Training process
        6. Common architectures
        7. Applications in different fields
        Please provide examples for each topic.""",
        "Provide a comprehensive guide to software architecture patterns..."
    ]
    
    # Process each question
    for question in questions:
        print(f"\nQuestion: \n{question}")
        print("=" * 80)
        print('\n')
        try:
            result = await orchestrator.process_request(question, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("gemini")
                    if content and content.success:
                        print("\nResponse:\n")
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
    asyncio.run(main()) 
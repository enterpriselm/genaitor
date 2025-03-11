import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import financial_agent 

async def main():
    print("\nInitializing Financial Analysis System...")
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"financial_agent": financial_agent},
        flows={
            "default_flow": Flow(agents=["financial_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Market Data: S&P 500, DXY, crude oil prices. Suggest strategies to minimize risk and maximize returns in high volatility.",
        "Customer Credit Data: Transaction history of clients with 60% probability of default. Predict and suggest improvements in credit granting process.",
        "Portfolio Data: Diverse portfolio of stocks, bonds, and real estate. Suggest adjustments to optimize risk-return balance.",
        "Transaction Data: Client transactions with unusual high-frequency trading. Detect potential fraudulent activities."
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("financial_agent")
                    if content and content.success:
                        print("\nSuggested Actions:")
                        print("-" * 80)
                        
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

import asyncio

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import optimization_agent

async def main():
    print("\nInitializing ML/DL Optimization System...")
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"optimization_agent": optimization_agent},
        flows={
            "default_flow": Flow(agents=["optimization_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Dataset: Image data for object classification task. Suggest the most suitable ML or DL model.",
        "Model: CNN. Dataset: Small image dataset. Tune hyperparameters to optimize model performance.",
        "Model: SVM. Dataset: Financial data. Evaluate model performance based on precision and recall.",
        "Model: Neural Network. Dataset: Large dataset with complex features. Suggest regularization techniques to prevent overfitting."
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("optimization_agent")
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

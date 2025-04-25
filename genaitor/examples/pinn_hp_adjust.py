import asyncio

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import pinn_tuning_agent

async def main():
    print("\nInitializing PINN Hyperparameter Tuning Demo...")
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"pinn_tuner": pinn_tuning_agent},
        flows={
            "default_flow": Flow(agents=["pinn_tuner"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Architecture: Fully connected feedforward network with 3 layers, 64 neurons per layer. Training parameters: Learning rate: 0.001, Batch size: 32, Epochs: 500. Review and suggest hyperparameter adjustments.",
        "Architecture: Convolutional neural network with 4 layers, kernel size 3x3, 128 neurons per layer. Training parameters: Learning rate: 0.0001, Batch size: 64, Epochs: 1000. Review and suggest hyperparameter adjustments.",
        "Architecture: 2-layer LSTM network with 256 neurons per layer. Training parameters: Learning rate: 0.01, Batch size: 128, Epochs: 200. Review and suggest hyperparameter adjustments."
    ]
    
    # Process each input and suggest adjustments
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("pinn_tuner")
                    if content and content.success:
                        print("\nSuggested Adjustments for Hyperparameters:")
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

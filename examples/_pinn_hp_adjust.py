import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

# Define custom task for suggesting adjustments to PINN hyperparameters
class PinnHyperparameterTuningTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the following input, suggest possible adjustments or modifications to the hyperparameters for better performance:
        
        Input: {input_data}
        
        Please review the architecture and training parameters and suggest any adjustments in the following areas:
        1. Learning rate
        2. Batch size
        3. Number of epochs
        4. Network architecture adjustments (e.g., number of layers, neurons per layer)
        5. Other relevant hyperparameters
        
        Format your response in a clear and actionable way, detailing any suggested changes.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "pinn_tuning"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing PINN Hyperparameter Tuning Demo...")
    test_keys = [os.getenv('API_KEY_1'), os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    pinn_tuning_task = PinnHyperparameterTuningTask(
        description="Suggest adjustments to hyperparameters for PINN training",
        goal="Suggest optimal hyperparameter settings for better PINN training performance",
        output_format="Suggested hyperparameter changes",
        llm_provider=provider
    )
    
    pinn_tuning_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[pinn_tuning_task],
        llm_provider=provider
    )
    
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

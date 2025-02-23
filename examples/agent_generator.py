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

# Define custom task for creating new agents
class AgentCreationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the following input, create a new task that an agent can perform:
        
        Input: {input_data}
        
        Please describe the new task, including:
        1. Description of the task
        2. The goal of the task
        3. Output format
        
        Format it in a way that it can be understood and performed by an agent.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "agent_creation"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Agent Creation Demo...")
    test_keys = [os.getenv('API_KEY_1'),os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    agent_creation_task = AgentCreationTask(
        description="Create a new agent based on user input",
        goal="Generate a new agent task description and goal",
        output_format="Task description, goal, and output format",
        llm_provider=provider
    )
    
    agent_creation = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[agent_creation_task],
        llm_provider=provider
    )
    
    # Setup orchestrator
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
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("creator")
                    if content and content.success:
                        print("\nGenerated Task for New Agent:")
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

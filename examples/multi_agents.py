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
    
# Define custom task
class LLMTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Input: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": self.description}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Multi-Agent System...")
    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    provider = GeminiProvider(gemini_config)
    
    # Define tasks
    qa_task = LLMTask(
        description="Question Answering",
        goal="Provide clear and accurate responses",
        output_format="Concise and informative",
        llm_provider=provider
    )
    
    summarization_task = LLMTask(
        description="Text Summarization",
        goal="Summarize lengthy content into key points",
        output_format="Bullet points or short paragraph",
        llm_provider=provider
    )
    
    # Create agents
    qa_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[qa_task],
        llm_provider=provider
    )
    summarization_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[summarization_task],
        llm_provider=provider
    )
    
    # Setup orchestrator with multiple agents
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
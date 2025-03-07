import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, TaskResult, Flow,
    ExecutionMode, AgentRole
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

class AutismSupportTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
        Please provide a response following the format:
        {self.output_format}

        Uses the Hyperfocus to improve the answer or the learning path for the answer
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "autism_support"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Autism Assistant...")
    test_keys = [os.getenv('API_KEY')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    autism_task = AutismSupportTask(
        description="Provide support and information related to the user query.",
        goal="Offer accurate and helpful responses using his hyperfocus as a way to improve the learning path.",
        output_format="Clear, concise response",
        llm_provider=provider
    )
    
    agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[autism_task],
        llm_provider=provider
    )
    
    orchestrator = Orchestrator(
        agents={"gemini": agent},
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
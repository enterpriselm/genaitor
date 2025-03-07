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

# Define custom task
class AutismSupportTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Question: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
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
    test_keys = [os.getenv('API_KEY_1'),os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    autism_task = AutismSupportTask(
        description="Provide support and information related to autism.",
        goal="Offer accurate and helpful responses to autism-related queries.",
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
    
    questions = [
        "What are the common signs of autism?",
        "How can I support a child with autism?",
        "What resources are available for autism awareness?"
    ]
    
    # Process each question
    for question in questions:
        print(f"\nQuestion: {question}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(question, flow_name='default_flow')
            
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
            break

if __name__ == "__main__":
    asyncio.run(main()) 
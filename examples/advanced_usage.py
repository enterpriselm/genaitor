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
class QuestionAnsweringTask(Task):
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
                metadata={"task_type": "qa"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Advanced Usage Demo...")
    test_keys = [os.getenv('API_KEY_1'),os.getenv('API_KEY_2')]
    
    # Set Gemini with token limits
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    # Create agent
    qa_task = QuestionAnsweringTask(
        description="Answer questions using Gemini",
        goal="Provide accurate and helpful answers",
        output_format="Clear, concise response",
        llm_provider=provider
    )
    
    agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[qa_task],
        llm_provider=provider
    )
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"gemini": agent},
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
                        
                        # Formata o texto removendo TaskResult
                        formatted_text = content.content.strip()
                        
                        # Remove marcadores Markdown desnecessários se desejar
                        formatted_text = formatted_text.replace("**", "")
                        
                        # Imprime com indentação adequada
                        for line in formatted_text.split('\n'):
                            if line.strip():  # Ignora linhas vazias
                                print(line)
                            else:
                                print()  # Mantém espaçamento entre parágrafos
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
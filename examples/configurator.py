import pandas as pd
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

class CarPurchaseTask(Task):
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
    print("\nInitializing Car Purchase System...")
    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    provider = GeminiProvider(gemini_config)
    
    preferences_task = CarPurchaseTask(
        description="Preferences Analysis",
        goal="Analyze the customer preferences and create a list of viable car models and options",
        output_format="JSON format with the analyzed preferences",
        llm_provider=provider
    )
    
    payment_task = CarPurchaseTask(
        description="Payment Calculation",
        goal="Calculate the payment conditions based on financing options and customer budget",
        output_format="JSON format with payment details and final amount",
        llm_provider=provider
    )
    
    proposal_task = CarPurchaseTask(
        description="Proposal Generation",
        goal="Generate a personalized proposal with the final price, payment options, and accessories included",
        output_format="Detailed proposal with car model, payment terms, accessories, and total cost",
        llm_provider=provider
    )

    review_task = CarPurchaseTask(
        description="Proposal Review",
        goal="Review the proposal to ensure it is clear, concise, and covers all customer needs",
        output_format="Clear and final version of the proposal",
        llm_provider=provider
    )
    
    preferences_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[preferences_task],
        llm_provider=provider
    )
    
    payment_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[payment_task],
        llm_provider=provider
    )
    
    proposal_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[proposal_task],
        llm_provider=provider
    )
    
    review_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[review_task],
        llm_provider=provider
    )

    orchestrator = Orchestrator(
        agents={"preferences_agent": preferences_agent, 
                "payment_agent": payment_agent,
                "proposal_agent": proposal_agent,
                "review_agent": review_agent},
        flows={
            "car_purchase_flow": Flow(agents=["preferences_agent", "payment_agent", "proposal_agent", "review_agent"], context_pass=[True, True, True, True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    customer_preferences = {
        "budget": 30000,
        "fuel_type": "Hybrid",
        "car_type": "SUV",
        "transmission": "Automatic",
        "seating_capacity": 5,
        "brand_preference": ["Toyota", "Honda"],
        "safety_rating": "5-star",
        "usage": "Daily commute and occasional road trips",
        "must_have_features": ["Adaptive Cruise Control", "Blind Spot Monitoring", "Apple CarPlay"]
    }
    
    try:
        result = await orchestrator.process_request(customer_preferences, flow_name='car_purchase_flow')
        
        if result["success"]:
            proposal_text = result['content']['review_agent'].content.strip()
            with open('examples/files/final_proposal.txt', 'w') as f:
                f.write(proposal_text)
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
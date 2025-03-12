import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import preferences_agent, payment_agent, proposal_agent, review_agent

async def main():
    print("\nInitializing Car Purchase System...")
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
            preferences_text = result['content']['preferences_agent'].content.strip()
            payment_text = result['content']['payment_agent'].content.strip()
            proposal_text = result['content']['proposal_agent'].content.strip()
            review_text = result['content']['review_agent'].content.strip()
            
            print("Preferences Text")
            print('\n')
            print(preferences_text)

            print("Payment Text")
            print('\n')
            print(payment_text)

            print("Proposal Text")
            print('\n')
            print(proposal_text)

            print("Review Text")
            print('\n')
            print(review_text)

            with open('examples/files/final_proposal.txt', 'w') as f:
                f.write(review_text)

            
            
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
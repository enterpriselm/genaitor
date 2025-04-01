import streamlit as st
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genaitor.core import Orchestrator, Flow, ExecutionMode
from src.genaitor.presets.agents import preferences_agent, payment_agent, proposal_agent, review_agent

def run_car_purchase(customer_preferences):
    async def main():
        orchestrator = Orchestrator(
            agents={
                "preferences_agent": preferences_agent,
                "payment_agent": payment_agent,
                "proposal_agent": proposal_agent,
                "review_agent": review_agent
            },
            flows={
                "car_purchase_flow": Flow(
                    agents=["preferences_agent", "payment_agent", "proposal_agent", "review_agent"],
                    context_pass=[True, True, True, True]
                )
            },
            mode=ExecutionMode.SEQUENTIAL
        )
        result = await orchestrator.process_request(customer_preferences, flow_name='car_purchase_flow')
        return result
    
    return asyncio.run(main())

# Streamlit UI
st.title("Car Purchase Assistant")

st.subheader("Enter Your Preferences:")

budget = st.number_input("Budget ($)", min_value=5000, max_value=100000, value=30000, step=1000)
fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Diesel", "Hybrid", "Electric"])
car_type = st.selectbox("Car Type", ["Sedan", "SUV", "Truck", "Coupe", "Hatchback"])
transmission = st.selectbox("Transmission", ["Automatic", "Manual"])
seating_capacity = st.number_input("Seating Capacity", min_value=2, max_value=8, value=5)
brand_preference = st.multiselect("Preferred Brands", ["Toyota", "Honda", "Ford", "BMW", "Tesla", "Mercedes"])
safety_rating = st.selectbox("Safety Rating", ["3-star", "4-star", "5-star"])
usage = st.text_input("Primary Usage", "Daily commute and occasional road trips")
must_have_features = st.multiselect("Must-Have Features", ["Adaptive Cruise Control", "Blind Spot Monitoring", "Apple CarPlay", "Android Auto", "Heated Seats"])

customer_preferences = {
    "budget": budget,
    "fuel_type": fuel_type,
    "car_type": car_type,
    "transmission": transmission,
    "seating_capacity": seating_capacity,
    "brand_preference": brand_preference,
    "safety_rating": safety_rating,
    "usage": usage,
    "must_have_features": must_have_features
}

if st.button("Find My Car"):
    with st.spinner("Processing..."):
        result = run_car_purchase(customer_preferences)
        
        if result["success"]:
            st.subheader("Recommendations:")
            st.write(f"**Preferences Analysis:** {result['content']['preferences_agent'].content.strip()}")
            st.write(f"**Payment Options:** {result['content']['payment_agent'].content.strip()}")
            st.write(f"**Proposed Cars:** {result['content']['proposal_agent'].content.strip()}")
            st.write(f"**Final Review:** {result['content']['review_agent'].content.strip()}")
        else:
            st.error(f"Error: {result['error']}")

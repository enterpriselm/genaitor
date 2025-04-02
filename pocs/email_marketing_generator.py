import streamlit as st
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genaitor.core import Orchestrator, Flow, ExecutionMode
from src.genaitor.presets.agents import research_agent, content_agent, optimization_agent, personalization_agent


def setup_orchestrator():
    return Orchestrator(
        agents={
            "research_agent": research_agent,
            "content_agent": content_agent,
            "optimization_agent": optimization_agent,
            "personalization_agent": personalization_agent
        },
        flows={
            "email_marketing_flow": Flow(
                agents=["research_agent", "content_agent", "optimization_agent", "personalization_agent"],
                context_pass=[True, True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

async def generate_email(campaign_details):
    orchestrator = setup_orchestrator()
    result = await orchestrator.process_request(
        {"campaign_details": campaign_details},
        flow_name='email_marketing_flow'
    )
    return result

st.title("📧 Marketing Email Generator")
st.write("Fill in the details below to generate an optimized campaign email.")

product = st.text_input("Product", "New Machine Learning course")
audience = st.text_input("Target Audience", "Tech professionals interested in AI")
goal = st.text_input("Goal", "Generate leads and increase conversions")

if st.button("Generate Email"):
    campaign_details = {
        "product": product,
        "audience": audience,
        "goal": goal
    }
    with st.spinner("Generating email..."):
        email_result = asyncio.run(generate_email(campaign_details))
        if email_result["success"]:
            st.subheader("✉️ Final Email:")
            st.markdown(email_result['content']['personalization_agent'].content.strip(), unsafe_allow_html=True)
        else:
            st.error(f"Error: {email_result['error']}")
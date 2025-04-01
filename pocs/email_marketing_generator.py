import streamlit as st
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genaitor.core import Orchestrator, Flow, ExecutionMode
from src.genaitor.presets.agents import research_agent, content_agent, optimization_agent, personalization_agent


# Configuração da Orquestração
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

st.title("📧 Gerador de E-mails de Marketing")
st.write("Preencha os detalhes abaixo para gerar um e-mail de campanha otimizado.")

product = st.text_input("Produto", "Novo curso de Machine Learning")
audience = st.text_input("Público-alvo", "Profissionais de tecnologia interessados em IA")
goal = st.text_input("Objetivo", "Gerar leads e aumentar conversões")

if st.button("Gerar E-mail"):
    campaign_details = {
        "product": product,
        "audience": audience,
        "goal": goal
    }
    with st.spinner("Gerando e-mail..."):
        email_result = asyncio.run(generate_email(campaign_details))
        if email_result["success"]:
            st.subheader("✉️ E-mail Final:")
            st.write(email_result['content'])
        else:
            st.error(f"Erro: {email_result['error']}")
import streamlit as st
import asyncio

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import autism_agent

def process_request(hyperfocus, question):
    input_data = f"Hyperfocus: {hyperfocus}\nQuestion: {question}"
    return asyncio.run(main(input_data))

async def main(input_data):
    orchestrator = Orchestrator(
        agents={"gemini": autism_agent},
        flows={
            "default_flow": Flow(agents=["gemini"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    try:
        result = await orchestrator.process_request(input_data, flow_name='default_flow')
        if result["success"]:
            content = result["content"].get("gemini")
            if content and content.success:
                return content.content.strip()
            else:
                return "Empty response received"
        else:
            return f"Error: {result['error']}"
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Autism Assistant")
st.write("Enter your hyperfocus topic and a question to receive a response.")

hyperfocus = st.text_input("Hyperfocus Topic", "Soccer")
question = st.text_area("Your Question", "What is Data Science?")

if st.button("Submit"):
    response = process_request(hyperfocus, question)
    st.write("### Response:")
    st.write(response)
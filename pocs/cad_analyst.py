import streamlit as st
import asyncio
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import problem_analysis_agent, numerical_analysis_agent, pinn_modeling_agent

async def process_problem(user_requirements, problem_description):
    orchestrator = Orchestrator(
        agents={"problem_analysis_agent": problem_analysis_agent,
                "numerical_modelling_agent": numerical_analysis_agent, 
                "pinn_modeling_agent": pinn_modeling_agent},
        flows={
            "problem_solving_flow": Flow(agents=["problem_analysis_agent", "numerical_modelling_agent", "pinn_modeling_agent"], context_pass=[True, True, True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    input_data = f"User Requirements: {user_requirements}\n\nProblem Description: {problem_description}"
    
    try:
        result = await orchestrator.process_request(input_data, flow_name='problem_solving_flow')
        
        if result["success"]:
            problem_analysis = result['content']['problem_analysis_agent'].content.strip()
            math_code = result['content']['numerical_modelling_agent'].content.strip()
            pinn_code = result['content']['pinn_modeling_agent'].content.strip()
            
            return problem_analysis, math_code, pinn_code
        else:
            return f"Error: {result['error']}", "", ""
    except Exception as e:
        return f"Error: {str(e)}", "", ""

st.title("FEM/FVM/FEA Problem Solver with PINN")

user_requirements = st.text_area("Enter user requirements:", "Solve a heat conduction problem in a 2D plate.")
problem_description = st.text_area("Enter problem description:", "Given the 2D heat conduction equation, choose the appropriate method (FEM/FVM/FEA), describe the methodology, and provide the solution with Python code.")

if st.button("Solve Problem"):
    with st.spinner("Processing..."):
        problem_analysis, math_code, pinn_code = asyncio.run(process_problem(user_requirements, problem_description))
    
    st.subheader("Problem Analysis")
    st.text(problem_analysis)
    
    st.subheader("Numerical Modelling Code")
    st.code(math_code, language='python')
    
    st.subheader("PINN Model Code")
    st.code(pinn_code, language='python')
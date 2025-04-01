import streamlit as st
import asyncio

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import equation_solver_agent, pinn_generation_agent, hyperparameter_optimization_agent, orchestrator_agent, validator_agent

def run_orchestration(user_input, execution_mode):
    async def main():
        if execution_mode == 'Sequential':
            orchestrator = Orchestrator(
                agents={
                    "solver_agent": equation_solver_agent,
                    "pinn_agent": pinn_generation_agent,
                    "optimizer_agent": hyperparameter_optimization_agent
                },
                flows={
                    "sequential_flow": Flow(agents=["solver_agent", "pinn_agent", "optimizer_agent"], context_pass=[True, True, True])
                },
                mode=ExecutionMode.SEQUENTIAL
            )
            result = await orchestrator._process_sequential(user_input, flow_name='sequential_flow')
        
        elif execution_mode == 'Parallel':
            orchestrator = Orchestrator(
                agents={
                    "solver_agent": equation_solver_agent,
                    "pinn_agent": pinn_generation_agent,
                    "optimizer_agent": hyperparameter_optimization_agent
                },
                flows={
                    "parallel_flow": Flow(agents=["solver_agent", "pinn_agent", "optimizer_agent"], context_pass=[True, True, True])
                },
                mode=ExecutionMode.PARALLEL
            )
            result = await orchestrator._process_parallel(user_input, flow_name='parallel_flow')
        
        elif execution_mode == 'Adaptative':
            orchestrator = Orchestrator(
                agents={
                    "solver_agent": equation_solver_agent,
                    "pinn_agent": pinn_generation_agent,
                    "optimizer_agent": hyperparameter_optimization_agent,
                    "orchestrator": orchestrator_agent,
                    "validator": validator_agent
                },
                flows={
                    "adaptive_flow": Flow(
                        agents=["solver_agent", "pinn_agent", "optimizer_agent"], 
                        context_pass=[True, True, True],
                        orchestrator_agent="orchestrator",
                        validator_agent="validatoxr"
                    )
                },
                mode=ExecutionMode.ADAPTIVE
            )
            result = await orchestrator._process_adaptative(user_input, flow_name='adaptive_flow')
        
        return result
    
    return asyncio.run(main())

# Streamlit UI
st.title("Multi-Agent PINN System")

example_inputs = [
    "Solve the Schrödinger equation for a quantum harmonic oscillator.",
    "Design a PINN to approximate the Navier-Stokes equations in 2D.",
    "Optimize the learning rate and activation functions for a PINN solving wave equations."
]

input_text = st.text_area("Enter your problem description:", example_inputs[0])
execution_mode = st.selectbox("Select Execution Mode", ["Sequential", "Parallel", "Adaptative"])

if st.button("Run Simulation"):
    with st.spinner("Processing..."):
        result = run_orchestration(input_text, execution_mode)
        
        st.subheader("Results:")
        for agent, answer in result["content"].items():
            st.write(f"**{agent.capitalize()}**: {answer.content}")

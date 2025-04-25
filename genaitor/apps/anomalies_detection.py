import streamlit as st
import pandas as pd
import asyncio

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import data_understanding_agent, statistics_agent, anomalies_detection_agent, data_analysis_agent

def process_file(file):
    df = pd.read_csv(file)
    return df

async def analyze_data(input_data):
    orchestrator = Orchestrator(
        agents={"data_understanding_agent": data_understanding_agent, 
                "statistics_agent": statistics_agent,
                "anomalies_detection_agent": anomalies_detection_agent,
                "data_analysis_agent": data_analysis_agent},
        flows={
            "anomalies_detection_flow": Flow(agents=["data_understanding_agent", "statistics_agent", "anomalies_detection_agent","data_analysis_agent"], context_pass=[True,True,True,True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    try:
        result = await orchestrator.process_request(input_data, flow_name='anomalies_detection_flow')
        
        if result["success"]:
            python_codes = result['content']['data_analysis_agent'].content.strip().split('```')
            for python_code in python_codes:
                if python_code.startswith('python'):
                    return python_code.partition('python')[2]
        else:
            return f"Error: {result['error']}"
    
    except Exception as e:
        return f"Error: {str(e)}"

st.title("Anomaly Detection System")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = process_file(uploaded_file)
    st.write("### Data Preview", df.head())
    
    if st.button("Run Analysis"):
        result = asyncio.run(analyze_data(df))
        st.code(result, language='python')

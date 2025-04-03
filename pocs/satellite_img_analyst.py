import streamlit as st
import rasterio
import numpy as np
import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genaitor.core import Orchestrator, Flow, ExecutionMode
from src.genaitor.presets.agents import (
    disaster_analysis_agent, agro_analysis_agent, ecological_analysis_agent,
    air_quality_analysis_agent, vegetation_analysis_agent, soil_moisture_analysis_agent
)

def extract_bands(img_path):
    """Extracts bands from a raster image."""
    image_band = {}
    with rasterio.open(img_path) as dataset:
        for i in range(1, dataset.count + 1):
            image_band[i] = dataset.read(i)
    return image_band

async def analyze_image(image_band):
    """Runs the analysis flow on the given image bands."""
    orchestrator = Orchestrator(
        agents={
            "Disaster Analysis": disaster_analysis_agent,
            "Agro Analysis": agro_analysis_agent,
            "Ecological Analysis": ecological_analysis_agent,
            "Air Quality Analysis": air_quality_analysis_agent,
            "Vegetation Analysis": vegetation_analysis_agent,
            "Soil Moisture Analysis": soil_moisture_analysis_agent
        },
        flows={
            "analysis_flow": Flow(
                agents=["Disaster Analysis", "Agro Analysis", "Ecological Analysis", "Air Quality Analysis", "Vegetation Analysis", "Soil Moisture Analysis"],
                context_pass=[True, True, True, True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    try:
        result = await orchestrator.process_request({"input_data": image_band}, flow_name='analysis_flow')
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

st.title("🌍 GeoTIFF Image Analysis")
uploaded_file = st.file_uploader("Upload a GeoTIFF file", type=["tif", "tiff"])

if uploaded_file is not None:
    temp_file_path = os.path.join("temp.tif")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    st.write("Extracting image bands...")
    image_band = extract_bands(temp_file_path)
    st.write(f"Extracted {len(image_band)} bands from the image.")
    
    if st.button("Run Analysis"):
        st.write("Analyzing image...")
        result = asyncio.run(analyze_image(image_band))
        
        if result["success"]:
            for agent, response in result["content"].items():
                st.subheader(f"{agent} Results")
                st.write(response.content.strip())
        else:
            st.error(f"Error: {result['error']}")
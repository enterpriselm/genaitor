import asyncio
import rasterio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import (
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

def main():
    image_band = extract_bands('genaitor\examples\files\S5P_OFFL_L1B_IR_UVN_20250423T033001_20250423T051131_38998_03_020101_20250423T065508.nc')
    print(f"Extracted {len(image_band)} bands from the image.")
    
    print("Analyzing image...")
    result = asyncio.run(analyze_image(image_band))
    
    if result["success"]:
        for agent, response in result["content"].items():
            print(f"{agent} Results")
            print(response.content.strip())
    else:
        print(f"Error: {result['error']}")
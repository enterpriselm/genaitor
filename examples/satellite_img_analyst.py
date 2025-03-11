import rasterio
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import disaster_analysis_agent, agro_analysis_agent, ecological_analysis_agent, air_quality_analysis_agent, vegetation_analysis_agent, soil_moisture_analysis_agent

async def main(img_path):

    image_band = {}
    for i in range(1, 14): 
        with rasterio.open(img_path) as dataset:
            try:
                image_band[i] = dataset.read(i)
            except:
                pass

    orchestrator = Orchestrator(
        agents={
            "Disaster Analysis": disaster_analysis_agent,
            "Agro Analysis": agro_analysis_agent,
            "Ecological Analysis": ecological_analysis_agent,
            "Air Quality Analysis": air_quality_analysis_agent,
            "Vegetation Analysis": vegetation_analysis_agent,
            "Soil Moisture Analysis": soil_moisture_analysis_agent},
        flows={
            "analysis_flow": Flow(
                agents=["Disaster Analysis", "Agro Analysis", "Ecological Analysis", "Air Quality Analysis", "Vegetation Analysis", "Soil Moisture Analysis"],
        context_pass=[True, True, True, True, True, True])},
        mode=ExecutionMode.SEQUENTIAL)
    
    try:
        result = await orchestrator.process_request({"input_data": image_band}, flow_name='analysis_flow')
        i=0
        if result["success"]:
            for agent in ["Disaster Analysis", "Agro Analysis", "Ecological Analysis", "Air Quality Analysis", "Vegetation Analysis", "Soil Moisture Analysis"]:
                content = result['content'][agent].content.strip()
                with open('examples/files/'+agent.lower().replace(' ','_')+'.txt','w') as f:
                        f.write(content)
        else:
            print(f"\nError: {result['error']}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main(img_path=r'examples\files\sample.tif'))
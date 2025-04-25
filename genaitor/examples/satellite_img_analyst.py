import matplotlib.pyplot as plt
import numpy as np
import asyncio
from netCDF4 import Dataset
from skimage.transform import resize

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import disaster_analysis_agent, agro_analysis_agent, ecological_analysis_agent, air_quality_analysis_agent, vegetation_analysis_agent, soil_moisture_analysis_agent

def extract_band_arrays(group, var_keywords=["irradiance", "radiance"]):
    for subgroup_name, subgroup in group.groups.items():
        result = extract_band_arrays(subgroup, var_keywords)
        if result is not None:
            return result

        for var_name, var in subgroup.variables.items():
            if any(key in var_name.lower() for key in var_keywords):
                return var[:]
    return None

def extract_bands(img_path):
    nc_file = Dataset(img_path, mode="r")
    
    band_arrays = []
    band_names = []

    for band_group_name in nc_file.groups:
        if band_group_name.startswith("BAND") and "IRRADIANCE" in band_group_name:
            band_group = nc_file.groups[band_group_name]
            arr = extract_band_arrays(band_group)
            if arr is not None:
                band_arrays.append(arr)
                band_names.append(band_group_name)

    return band_arrays, band_names

async def main(image_band, band_name):
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
        result = await orchestrator.process_request({"input_data": f"Band Name: {band_name}\n\nBand Array: {image_band}"}, flow_name='analysis_flow')
        return result
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    band_arrays, band_names = extract_bands('examples\files\S5P_OFFL_L1B_IR_UVN_20250423T033001_20250423T051131_38998_03_020101_20250423T065508.nc')
        
    for i in range(0, len(band_names)):
        result = asyncio.run(main(band_arrays[i], band_names[i]))
    
        if result["success"]:
            for agent, response in result["content"].items():
                print(f"{agent} Results")
                print(response.content.strip())
        else:
            print(f"Error: {result['error']}")
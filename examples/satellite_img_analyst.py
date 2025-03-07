import rasterio
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

class DisasterAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Environmental Disaster Analysis",
            goal="Detect environmental disasters such as floods, wildfires, or landslides using spectral bands",
            output_format="Detailed report of potential environmental disasters detected in the selected bands"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a detailed report on potential environmental disasters detected, such as floods, wildfires, or landslides.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

class AgroAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Agricultural Analysis",
            goal="Monitor crop health and detect signs of water stress or pests using spectral bands",
            output_format="Crop health report, highlighting signs of water stress or pest infestation"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on crop health, highlighting signs of water stress or pest infestation.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

class EcologicalAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Ecological Analysis",
            goal="Study local vegetation and ecosystems to detect signs of environmental degradation",
            output_format="Report on signs of environmental degradation in local ecosystems"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on signs of environmental degradation in local ecosystems.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

class AirQualityAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Air Quality Analysis",
            goal="Detect air pollution, such as smoke or other contaminants in the spectral bands",
            output_format="Report on areas with detected air pollution"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on areas with detected air pollution, such as smoke or other contaminants.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

class VegetationAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Vegetation Analysis",
            goal="Detect deforestation or changes in vegetation using spectral bands",
            output_format="Report on deforestation or changes in vegetation detected"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on deforestation or changes in vegetation detected.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

class SoilMoistureAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Soil Moisture Analysis",
            goal="Study soil moisture and identify areas prone to drought or excess water",
            output_format="Report on soil moisture and regions prone to drought or excess water"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on soil moisture and areas prone to drought or excess water.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

async def main(img_path):

    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
            api_keys=test_keys,
            temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    llm_provider = GeminiProvider(gemini_config)
    
    disaster_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[DisasterAnalysisTask(llm_provider)],
        llm_provider=llm_provider
    )

    agro_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[AgroAnalysisTask(llm_provider)],
        llm_provider=llm_provider
    )

    ecological_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[EcologicalAnalysisTask(llm_provider)],
        llm_provider=llm_provider
    )

    air_quality_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[AirQualityAnalysisTask(llm_provider)],
        llm_provider=llm_provider
    )

    vegetation_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[VegetationAnalysisTask(llm_provider)],
        llm_provider=llm_provider
    )

    soil_moisture_analysis_agent = Agent(
        role=AgentRole.CUSTOM,
        tasks=[SoilMoistureAnalysisTask(llm_provider)],
        llm_provider=llm_provider
    )

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
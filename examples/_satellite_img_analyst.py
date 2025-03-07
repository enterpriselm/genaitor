import rasterio
import numpy as np
import matplotlib.pyplot as plt
import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

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

def process_satellite_img(img_path, agent):
    agent_mapping = {
        "Disaster Analysis": {
            "agent": disaster_analysis_agent,
            "spectral_bands": [2, 3, 4, 8, 11]
        },
        "Agro Analysis": {
            "agent": agro_analysis_agent,
            "spectral_bands": [3, 4, 8, 11]
        },
        "Ecological Analysis": {
            "agent": ecological_analysis_agent,
            "spectral_bands": [2, 3, 4, 8]
        },
        "Air Quality Analysis": {
            "agent": air_quality_analysis_agent,
            "spectral_bands": [1, 9, 10]
        },
        "Vegetation Analysis": {
            "agent": vegetation_analysis_agent,
            "spectral_bands": [4, 5, 6, 7, 8]
        },
        "Soil Moisture Analysis": {
            "agent": soil_moisture_analysis_agent,
            "spectral_bands": [9, 12, 13]
        }
    }
    
    with rasterio.open(img_path) as dataset:
        image_band = {}
        for band in agent_mapping[agent]["spectral_bands"]:
            image_band[band] = dataset.read(band)

    orchestrator = Orchestrator(
        agents={
            agent: agent_mapping[agent]['agent']},
        flows={
            "analysis_flow": Flow(
                agents=[
            agent],
        context_pass=[True])},
        mode=ExecutionMode.SEQUENTIAL)
    
    result_process = orchestrator.process_request(
        {"input_data": image_band},
        flow_name="analysis_flow"
    )
    result = asyncio.run(result_process)
    
process_satellite_img('caminho/para/imagem_sentinel.tif')

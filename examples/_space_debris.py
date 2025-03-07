import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Task, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')


class SpaceDebrisDetectionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following satellite images and radar data, identify and categorize space debris:
        
        Input: {input_data}
        
        Please classify the debris into categories such as size, material, and potential risk.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "space_debris_detection"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

class TrajectoryPredictionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the current trajectory data and environmental conditions, predict the future path of space debris:
        
        Input: {input_data}
        
        Please provide a predicted collision risk, including the likelihood and timing of potential impacts.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "trajectory_prediction"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

class MitigationStrategyTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the predicted trajectories and collision risks, suggest mitigation strategies to remove or avoid space debris:
        
        Input: {input_data}
        
        Please provide recommendations such as satellite maneuvers, debris removal technologies, or orbital changes.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "mitigation_strategy"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

class EnvironmentalImpactAnalysisTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the current state of space debris and policies in place, analyze the environmental and political impacts:
        
        Input: {input_data}
        
        Provide recommendations for policy changes or technological advances to mitigate space debris in the long term.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "environmental_impact_analysis"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def space_debris_agent():
    # Carregar dados
    input_data_detection = "Radar data from satellite #1 showing debris at 300 km altitude"
    input_data_prediction = "Current debris trajectory data for object #53"
    input_data_mitigation = "Collision risk prediction for satellite X and debris object Y"
    input_data_impact_analysis = "Environmental impact of current space debris levels"

    # Inicializar os agentes com suas tarefas
    detection_task = SpaceDebrisDetectionTask("Detect space debris", "Detect and classify debris", "Text", llm_provider)
    prediction_task = TrajectoryPredictionTask("Predict debris trajectories", "Predict future trajectories", "Text", llm_provider)
    mitigation_task = MitigationStrategyTask("Propose mitigation strategies", "Suggest strategies to avoid debris collisions", "Text", llm_provider)
    impact_analysis_task = EnvironmentalImpactAnalysisTask("Analyze environmental impact", "Analyze impact of space debris", "Text", llm_provider)
    
    # Executar tarefas
    detection_result = await detection_task.execute(input_data_detection)
    prediction_result = await prediction_task.execute(input_data_prediction)
    mitigation_result = await mitigation_task.execute(input_data_mitigation)
    impact_analysis_result = await impact_analysis_task.execute(input_data_impact_analysis)
    
    # Imprimir resultados
    print(detection_result.content)
    print(prediction_result.content)
    print(mitigation_result.content)
    print(impact_analysis_result.content)

if __name__ == "__main__":
    asyncio.run(space_debris_agent())

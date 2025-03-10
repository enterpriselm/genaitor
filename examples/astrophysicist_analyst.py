import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
load_dotenv('.env')

class GeneralTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": self.description}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    print("\nInitializing Astrophysicist Analyst Pipeline...")
    test_keys = [os.getenv('API_KEY')]

    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=10000
    )
    provider = GeminiProvider(gemini_config)

    motion_analysis_task = GeneralTask(
        description="Analyze stellar motion data to identify stars with high velocity moving toward the center of the Milky Way.",
        goal="Identify stars exhibiting high proper motion and radial velocity directed toward the galactic center.",
        output_format="List of stars with high velocity toward the center of the Milky Way.",
        llm_provider=provider
    )

    photometric_classification_task = GeneralTask(
        description="Classify stars based on their BP and RP photometric data, distinguishing types like red dwarfs, giants, and others.",
        goal="Identify different stellar types based on their temperature and composition using Gaia's photometric data.",
        output_format="Classification of stars into categories such as red dwarfs, giants, etc.",
        llm_provider=provider
    )

    galactic_dynamics_task = GeneralTask(
        description="Analyze stellar velocity and proper motion data to understand the movement of stars relative to the Sun and the rotation of the Milky Way.",
        goal="Study stellar velocity distributions to investigate the dynamics of the Milky Way, including star movements relative to the galactic center.",
        output_format="Understanding of the rotation of the Milky Way and stellar movement patterns.",
        llm_provider=provider
    )

    galactic_structure_task = GeneralTask(
        description="Study the distribution of stars in the Milky Way and identify structures such as spiral arms and star-forming regions.",
        goal="Map stellar distribution and identify spiral arm patterns and star-forming regions in the Milky Way.",
        output_format="Maps of stellar distribution and identification of galactic structures.",
        llm_provider=provider
    )

    stellar_variability_task = GeneralTask(
        description="Detect variable stars and explosive events such as supernovae based on their light curves.",
        goal="Identify variable stars and supernova events by analyzing changes in their brightness over time.",
        output_format="Classification of variable stars and detection of explosive events.",
        llm_provider=provider
    )

    chemical_composition_task = GeneralTask(
        description="Analyze the spectroscopic data of stars in the halo of the Milky Way to determine their chemical composition.",
        goal="Study the metallicity and chemical patterns of stars in the Milky Way’s halo to understand its formation and evolution.",
        output_format="Determination of stellar chemical composition and insights into galactic evolution.",
        llm_provider=provider
    )

    exoplanet_detection_task = GeneralTask(
        description="Identify stars with potential exoplanets based on astrometric data and radial velocity measurements.",
        goal="Use astrometric data to detect stars with exoplanets and calculate their orbits.",
        output_format="List of stars with exoplanet candidates and calculated orbital parameters.",
        llm_provider=provider
    )

    binary_system_analysis_task = GeneralTask(
        description="Study the dynamics of binary star systems using motion and distance data to infer their properties and impact on star formation.",
        goal="Analyze binary star system dynamics and their implications for star formation processes.",
        output_format="Classification and properties of binary star systems and insights into their role in star formation.",
        llm_provider=provider
    )

    space_mission_planning_task = GeneralTask(
        description="Optimize spacecraft trajectories based on planetary orbits and celestial body positions.",
        goal="Calculate optimized trajectories for space exploration missions using planetary orbital data.",
        output_format="Optimized trajectory plan and mission parameters.",
        llm_provider=provider
    )

    scientific_discovery_task = GeneralTask(
        description="Suggest new research approaches based on the latest scientific publications in a given field of physics.",
        goal="Identify emerging trends and suggest new hypotheses for future research based on the latest publications.",
        output_format="Suggested hypotheses and research approaches for future investigation.",
        llm_provider=provider
    )

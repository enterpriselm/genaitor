from src.core import Agent, Task, AgentRole, Orchestrator, Flow
from src.llm import LLMProvider
from examples.advanced_usage import QuestionAnsweringTask
from examples.autism_assistant import AutismSupportTask
from examples.forecasting_builder import ForecastingTask
from examples.pinn_builder import PinnTask
from examples.pinn_trainer import PinnTrainerTask

# Define custom tasks
class CustomQuestionAnsweringTask(QuestionAnsweringTask):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            description="Answer questions using Gemini",
            goal="Provide accurate and helpful answers",
            output_format="Clear, concise response",
            llm_provider=llm_provider
        )

class CustomAutismSupportTask(AutismSupportTask):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            description="Provide support and information related to autism.",
            goal="Offer accurate and helpful responses to autism-related queries.",
            output_format="Clear, concise response",
            llm_provider=llm_provider
        )

class CustomForecastingTask(ForecastingTask):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            description="Generate forecasts based on input data.",
            goal="Provide accurate and helpful forecasts.",
            output_format="Clear, concise forecast",
            llm_provider=llm_provider
        )

class CustomPinnTask(PinnTask):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            description="Build a Physics-Informed Neural Network (PINN) based on input data.",
            goal="Provide accurate and helpful PINN configurations.",
            output_format="Clear, concise PINN configuration",
            llm_provider=llm_provider
        )

class CustomPinnTrainerTask(PinnTrainerTask):
    def __init__(self, llm_provider: LLMProvider):
        super().__init__(
            description="Generate Python code to train a Physics-Informed Neural Network (PINN).",
            goal="Provide a clear and functional code snippet for training the PINN.",
            output_format="Python code for training the PINN",
            llm_provider=llm_provider
        )

def create_agents(llm_provider: LLMProvider):
    """Create and return a dictionary of agents with their respective tasks."""
    agents = {
        "qa_agent": Agent(role=AgentRole.SPECIALIST, tasks=[CustomQuestionAnsweringTask(llm_provider)], llm_provider=llm_provider),
        "autism_agent": Agent(role=AgentRole.SPECIALIST, tasks=[CustomAutismSupportTask(llm_provider)], llm_provider=llm_provider),
        "forecasting_agent": Agent(role=AgentRole.SPECIALIST, tasks=[CustomForecastingTask(llm_provider)], llm_provider=llm_provider),
        "pinn_agent": Agent(role=AgentRole.SPECIALIST, tasks=[CustomPinnTask(llm_provider)], llm_provider=llm_provider),
        "pinn_trainer_agent": Agent(role=AgentRole.SPECIALIST, tasks=[CustomPinnTrainerTask(llm_provider)], llm_provider=llm_provider),
    }
    return agents

def create_flows():
    """Define and return a dictionary of flows for orchestrators."""
    return {
        "default_flow": Flow(agents=["qa_agent", "autism_agent"], context_pass=[True, False]),
        "pinn_flow": Flow(agents=["pinn_agent", "pinn_trainer_agent"], context_pass=[True, True]),
        # Add more flows as needed
    }
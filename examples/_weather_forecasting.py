import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Configuração do LLM Provider
llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

# 1. Simulação Climática
class ClimateSimulationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Climate Simulation",
            "Simulate climate variables (temperature, precipitation, humidity) based on historical and current data",
            "JSON format with simulated climate data"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input: {input_data}

        Simulate the climate variables such as temperature, precipitation, humidity, etc., based on historical data and current conditions. Provide simulated data for specific future periods.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "climate_simulation"})

# 2. Análise de Resultados Climáticos
class ClimateResultAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Climate Result Analysis",
            "Analyze the simulated climate data and extract key patterns and trends",
            "JSON format with insights and trends"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Simulated Data: {input_data}

        Analyze the climate simulation data and extract key patterns such as temperature trends, precipitation variations, and potential climate shifts.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "climate_result_analysis"})

# 3. Previsão Ambiental
class EnvironmentalPredictionTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Environmental Prediction",
            "Generate predictions for future environmental conditions based on simulated data",
            "JSON format with predicted future conditions"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Climate Simulation Results: {input_data}

        Based on the climate simulation results, generate predictions for future environmental conditions, including potential climate events, extreme weather, and other environmental impacts.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "environmental_prediction"})

# 4. Geração de Relatório Climático
class ClimateReportGenerationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Climate Report Generation",
            "Generate a detailed report summarizing the climate simulation results, trends, and predictions",
            "Final report in text format"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Environmental Predictions: {input_data}

        Generate a detailed climate report summarizing the simulation results, key trends, predictions for future climate conditions, and recommendations for action in response to potential changes.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "climate_report_generation"})

# Criando Agentes
climate_simulation_agent = Agent(
    role=AgentRole.SIMULATION_AGENT,
    tasks=[ClimateSimulationTask(llm_provider)],
    llm_provider=llm_provider
)

climate_result_analysis_agent = Agent(
    role=AgentRole.ANALYST,
    tasks=[ClimateResultAnalysisTask(llm_provider)],
    llm_provider=llm_provider
)

environmental_prediction_agent = Agent(
    role=AgentRole.PREDICTOR,
    tasks=[EnvironmentalPredictionTask(llm_provider)],
    llm_provider=llm_provider
)

climate_report_generation_agent = Agent(
    role=AgentRole.REPORT_GENERATOR,
    tasks=[ClimateReportGenerationTask(llm_provider)],
    llm_provider=llm_provider
)

# Criando o Orquestrador
orchestrator = Orchestrator(
    agents={
        "climate_simulation_agent": climate_simulation_agent,
        "climate_result_analysis_agent": climate_result_analysis_agent,
        "environmental_prediction_agent": environmental_prediction_agent,
        "climate_report_generation_agent": climate_report_generation_agent
    },
    flows={
        "climate_simulation_flow": Flow(
            agents=["climate_simulation_agent", "climate_result_analysis_agent", "environmental_prediction_agent", "climate_report_generation_agent"],
            context_pass=[True, True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

# Executando o fluxo
result_process = orchestrator.process_request(
    {"climate_data": "Historical climate data", "simulation_conditions": "Specific climate simulation conditions"},
    flow_name="climate_simulation_flow"
)
result = asyncio.run(result_process)

print(result)

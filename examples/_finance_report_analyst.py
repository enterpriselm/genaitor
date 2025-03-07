import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

class DataProcessingTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Data Processing", "Clean and preprocess financial data", "Structured JSON format")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Raw Financial Data: {input_data}

        Please clean and structure the data in JSON format.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "data_processing"})

class FinancialAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Financial Analysis", "Compute key financial metrics", "Revenue, Profit, Loss, Growth Rate")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Processed Financial Data: {input_data}

        Please compute key metrics including revenue, profit, loss, and growth rate.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "financial_analysis"})

class ReportGenerationTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Report Generation", "Generate a structured quarterly report", "Formatted PDF or Markdown")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Financial Metrics: {input_data}

        Please generate a structured quarterly report including an executive summary, detailed analysis, and key takeaways.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "report_generation"})

class DataVisualizationTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Data Visualization", "Create visual charts for financial insights", "Graphs and charts")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Financial Report Data: {input_data}

        Please generate visual insights in the form of graphs and charts.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "data_visualization"})

data_processing_agent = Agent(
    role=AgentRole.ENGINEER,
    tasks=[DataProcessingTask(llm_provider)],
    llm_provider=llm_provider
)

financial_analysis_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[FinancialAnalysisTask(llm_provider)],
    llm_provider=llm_provider
)

report_generation_agent = Agent(
    role=AgentRole.REFINER,
    tasks=[ReportGenerationTask(llm_provider)],
    llm_provider=llm_provider
)

visualization_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[DataVisualizationTask(llm_provider)],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={
        "data_processing_agent": data_processing_agent,
        "financial_analysis_agent": financial_analysis_agent,
        "report_generation_agent": report_generation_agent,
        "visualization_agent": visualization_agent
    },
    flows={
        "quarter_report_flow": Flow(
            agents=["data_processing_agent", "financial_analysis_agent", "report_generation_agent", "visualization_agent"],
            context_pass=[True, True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

result_process = orchestrator.process_request(
    "Quarterly financial data for Company XYZ (Q1 2025)",
    flow_name="quarter_report_flow"
)
result = asyncio.run(result_process)

print(result)

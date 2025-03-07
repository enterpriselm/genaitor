import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

class DataExtractionTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Data Extraction", "Extract relevant information from resumes", "JSON format with Name, Experience, Skills, Education")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Resume Text: {input_data}

        Please extract structured data including Name, Contact, Experience, Skills, Education, and Certifications in JSON format.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "data_extraction"})

class SkillMatchingTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Skill Matching", "Match extracted skills with job requirements", "Matched and missing skills in JSON format")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Extracted Resume Data: {input_data}

        Please compare the candidate’s skills with the job requirements and list matched skills, missing skills, and proficiency levels.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "skill_matching"})

class CVScoringTask(Task):
    def __init__(self, llm_provider):
        super().__init__("CV Scoring", "Score the resume based on experience, skills, and education", "Score out of 100")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Candidate Profile: {input_data}

        Please assign a score out of 100 based on relevance, experience, and qualifications.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "cv_scoring"})

class ReportGenerationTask(Task):
    def __init__(self, llm_provider):
        super().__init__("Report Generation", "Generate a structured candidate evaluation report", "PDF or Markdown format")
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        CV Analysis Data: {input_data}

        Please generate a structured report including candidate summary, skill match, score, and recommendations.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "report_generation"})

data_extraction_agent = Agent(
    role=AgentRole.ENGINEER,
    tasks=[DataExtractionTask(llm_provider)],
    llm_provider=llm_provider
)

skill_matching_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[SkillMatchingTask(llm_provider)],
    llm_provider=llm_provider
)

cv_scoring_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[CVScoringTask(llm_provider)],
    llm_provider=llm_provider
)

report_generation_agent = Agent(
    role=AgentRole.CUSTOM,
    tasks=[ReportGenerationTask(llm_provider)],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={
        "data_extraction_agent": data_extraction_agent,
        "skill_matching_agent": skill_matching_agent,
        "cv_scoring_agent": cv_scoring_agent,
        "report_generation_agent": report_generation_agent
    },
    flows={
        "cv_analysis_flow": Flow(
            agents=["data_extraction_agent", "skill_matching_agent", "cv_scoring_agent", "report_generation_agent"],
            context_pass=[True, True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

result_process = orchestrator.process_request(
    "John Doe's resume for a Data Scientist position",
    flow_name="cv_analysis_flow"
)
result = asyncio.run(result_process)

print(result)

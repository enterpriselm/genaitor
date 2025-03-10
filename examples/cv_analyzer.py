import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

import os
import sys
import json
import asyncio
from dotenv import load_dotenv
from typing import Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

load_dotenv('.env')

class CVTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: Dict[str, Any]) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input:
        {json.dumps(input_data, indent=4)}

        Provide the response in the following format:
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
    print("\n🚀 Inicializando sistema de análise de currículos...")

    test_keys = [os.getenv('API_KEY')]

    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=5000
    )
    provider = GeminiProvider(gemini_config)

    # Definição das tarefas
    data_extraction = CVTask(
        description="Data Extraction",
        goal="Extract relevant information from resumes",
        output_format="JSON format with Name, Experience, Skills, Education, and Certifications",
        llm_provider=provider
    )

    skill_matching = CVTask(
        description="Skill Matching",
        goal="Match extracted skills with job requirements",
        output_format="Matched and missing skills in JSON format",
        llm_provider=provider
    )

    cv_scoring = CVTask(
        description="CV Scoring",
        goal="Score the resume based on experience, skills, and education",
        output_format="Score out of 100",
        llm_provider=provider
    )

    report_generation = CVTask(
        description="Report Generation",
        goal="Generate a structured candidate evaluation report",
        output_format="PDF or Markdown format",
        llm_provider=provider
    )

    # Definição dos agentes
    extraction_agent = Agent(
        role=AgentRole.ANALYST,
        tasks=[data_extraction],
        llm_provider=provider
    )

    matching_agent = Agent(
        role=AgentRole.EVALUATOR,
        tasks=[skill_matching],
        llm_provider=provider
    )

    scoring_agent = Agent(
        role=AgentRole.SCORER,
        tasks=[cv_scoring],
        llm_provider=provider
    )

    report_agent = Agent(
        role=AgentRole.REPORTER,
        tasks=[report_generation],
        llm_provider=provider
    )

    # Orquestração do fluxo de análise de currículos
    orchestrator = Orchestrator(
        agents={
            "extraction_agent": extraction_agent,
            "matching_agent": matching_agent,
            "scoring_agent": scoring_agent,
            "report_agent": report_agent
        },
        flows={
            "cv_analysis_flow": Flow(
                agents=["extraction_agent", "matching_agent", "scoring_agent", "report_agent"],
                context_pass=[True, True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    # Exemplo de currículo de teste
    resume_text = """
    John Doe
    Email: johndoe@email.com | Phone: (123) 456-7890 | LinkedIn: linkedin.com/in/johndoe
    Experience:
      - Data Scientist at AI Corp (2019-Present): Developed ML models for financial forecasting.
      - Software Engineer at Tech Solutions (2015-2019): Built scalable cloud applications.
    Skills: Python, Machine Learning, Data Analysis, SQL, AWS, TensorFlow
    Education: MSc in Computer Science, Stanford University (2015)
    Certifications: AWS Certified Machine Learning Specialist
    """

    job_requirements = {
        "title": "Senior Data Scientist",
        "required_skills": ["Python", "Machine Learning", "Deep Learning", "SQL", "Cloud Computing"],
        "preferred_skills": ["NLP", "Big Data", "Kubernetes"]
    }

    print("\n🔍 Extraindo dados do currículo...")

    try:
        result = await orchestrator.process_request(
            {"resume_text": resume_text, "job_requirements": job_requirements},
            flow_name='cv_analysis_flow'
        )

        if result["success"]:
            print("\n📊 Relatório final gerado:")
            print(result['content'])
        else:
            print(f"\n❌ Erro: {result['error']}")

    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

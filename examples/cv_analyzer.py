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
    Orchestrator, Flow, ExecutionMode
)
from presets.agents import extraction_agent, matching_agent, scoring_agent, report_agent

async def main():
    print("\n🚀 Inicializando sistema de análise de currículos...")
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

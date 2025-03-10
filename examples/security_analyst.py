import os
import asyncio
import sys
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Load environment variables
load_dotenv('.env')

class SecurityTask(Task):
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

def scrape_security_content(url: str) -> str:
    """Scrapes security-related elements from a web page."""
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract scripts, input fields, and inline event handlers
        scripts = [script.string for script in soup.find_all("script") if script.string]
        inputs = [str(input_tag) for input_tag in soup.find_all("input")]
        event_handlers = [
            tag.attrs[attr] for tag in soup.find_all(True)
            for attr in tag.attrs if attr.startswith("on")
        ]

        security_data = {
            "scripts": scripts,
            "input_fields": inputs,
            "event_handlers": event_handlers
        }

        return json.dumps(security_data, indent=4)

    except Exception as e:
        return json.dumps({"error": str(e)})

async def main():
    print("\nInitializing Security Analysis System...")

    test_keys = [os.getenv('API_KEY')]

    # Configure Gemini LLM
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=5000
    )
    provider = GeminiProvider(gemini_config)

    # Define tasks
    security_scraping = SecurityTask(
        description="Security Scraping",
        goal="Perform security-related scraping on the specified URL to extract potential vulnerabilities",
        output_format="HTML content with security-related elements",
        llm_provider=provider
    )

    vulnerability_analysis = SecurityTask(
        description="Vulnerability Analysis",
        goal="Analyze collected data for potential vulnerabilities like SQL Injection, XSS, etc.",
        output_format="List of vulnerabilities detected",
        llm_provider=provider
    )

    security_report = SecurityTask(
        description="Security Report",
        goal="Generate a detailed security report based on the analysis results",
        output_format="Formatted security report",
        llm_provider=provider
    )

    # Create agents
    scraping_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[security_scraping],
        llm_provider=provider
    )

    analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[vulnerability_analysis],
        llm_provider=provider
    )

    report_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[security_report],
        llm_provider=provider
    )

    # Orchestrate tasks
    orchestrator = Orchestrator(
        agents={
            "scraping_agent": scraping_agent,
            "analysis_agent": analysis_agent,
            "report_agent": report_agent
        },
        flows={
            "security_analysis_flow": Flow(
                agents=["scraping_agent", "analysis_agent", "report_agent"],
                context_pass=[True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    target_url = "http://example.com"

    print(f"\nScraping security content from: {target_url}")

    scraped_data = scrape_security_content(target_url)

    print("\nStarting security analysis...")

    try:
        result = await orchestrator.process_request(scraped_data, flow_name='security_analysis_flow')

        if result["success"]:
            print("\nSecurity Analysis Results:")
            print(json.dumps(result['content'], indent=4))
        else:
            print(f"\nError: {result['error']}")

    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())

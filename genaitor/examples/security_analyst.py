import os
import asyncio
import sys
import requests
import json
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Orchestrator, Flow, ExecutionMode
from presets.agents import scraping_agent, analysis_agent, report_agent

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

    target_url = "http://google.com"

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

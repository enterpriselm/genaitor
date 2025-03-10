import requests
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

class WebScraping(Task):
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
    print("\nInitializing Web Scraping System...")
    test_keys = [os.getenv('API_KEY')]
    
    # Set up Gemini configuration
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=5000
    )
    provider = GeminiProvider(gemini_config)
    
    html_analysis_task = WebScraping(
        description="Find Necessary Data",
        goal="Retrieve all information about which HTML part is the information needed",
        output_format="Concise and informative",
        llm_provider=provider
    )
    
    scraper_generation = WebScraping(
        description="Code Generator",
        goal="Based on the HTML struture and on where the data is, create a python code to scrape that data and store in a file.",
        output_format="Complete, concize and documentated python code",
        llm_provider=provider
    )
    
    html_analysis_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[html_analysis_task],
        llm_provider=provider
    )
    scraper_generation_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[scraper_generation],
        llm_provider=provider
    )

    orchestrator = Orchestrator(
        agents={"html_analysis_agent": html_analysis_agent, 
                "scraper_generation_agent": scraper_generation_agent},
        flows={
            "web_scraping_flow": Flow(agents=["html_analysis_agent","scraper_generation_agent"], context_pass=[True,True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    html = BeautifulSoup(requests.get('https://scholar.google.com/scholar?hl=pt-BR&as_sdt=0%2C5&q=physics+informed+neural+networks&btnG=&oq=physics+informed+').content,'html.parser')
    html_text = " ".join(html.get_text().split())
    user_requirements = "All paper links for all pages"
    input_data = f"User Requirements:{user_requirements}/n/nHTML: {html_text}"
    print("Starting scraper generaton for the Guardian to get news titles.")
    try:
        result = await orchestrator.process_request(input_data, flow_name='web_scraping_flow')
        i=0
        if result["success"]:
            python_codes = result['content']['scraper_generation_agent'].content.strip().split('```')
            for python_code in python_codes:
                if python_code.startswith('python'):
                    i+=1
                    filename=f'examples/files/scraper_generation_{i}.py'
                    with open(filename,'w') as f:
                        f.write(python_code.partition('python')[2])
        else:
            print(f"\nError: {result['error']}")
    
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
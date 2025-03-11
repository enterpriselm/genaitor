import requests
from bs4 import BeautifulSoup
import pandas as pd
import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
from presets.agents import html_analysis_agent, scraper_generation_agent

async def main():
    print("\nInitializing Web Scraping System...")
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
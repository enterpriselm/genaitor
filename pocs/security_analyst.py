import streamlit as st
import requests
import json
from bs4 import BeautifulSoup
import asyncio

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import Orchestrator, Flow, ExecutionMode
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

async def analyze_security(scraped_data):
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
    
    result = await orchestrator.process_request(scraped_data, flow_name='security_analysis_flow')
    return result

st.title("Web Security Analysis Tool")
url = st.text_input("Enter the URL to analyze:")

if st.button("Analyze"):
    if url:
        st.write(f"Scraping security content from: {url}")
        scraped_data = scrape_security_content(url)
        
        st.write("Starting security analysis...")
        
        result = asyncio.run(analyze_security(scraped_data))
        
        if result["success"]:
            st.write("Security Analysis Results:")
            st.json(result['content'])
        else:
            st.error(f"Error: {result['error']}")
    else:
        st.warning("Please enter a valid URL.")
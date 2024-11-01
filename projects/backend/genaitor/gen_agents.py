import requests
import json

def generate_agent(prompt):
    url = "http://127.0.0.1:5000/generate-agent"
    headers = {
        "X-API-Key": "E3WYK61nzXrGQ4fAdLgY6B53owyneVuR"
    }
    payload = {
        "user_query": prompt
    }
    response = requests.post(url, json=payload)#, headers=headers)
    return response.json()['ai_agent_prompt'].replace('</s>','')

def save_agent_structure(prompt, agent_path):
    agent_prompt = generate_agent(prompt)
    data = {
        "prompt": prompt,
        "agent_prompt": agent_prompt
    }
    with open(f'agents_prompts/{agent_path}.json', 'w') as f:
        f.write(json.dumps(data))

list_of_agents = [
    {
        "name": "answers_crafter",
        "prompt": "I need an AI Agent that does this: Based on an input to a llm and the output generated, you will validate if the output is correct or not. If is not correct, you will generate a new output"
    },
    {
        "name": "code_reviewer",
        "prompt": "I need an AI Agent that does this: You will receive a code snippet and must return a review of the code, based on 'python', 'html', 'css', 'javascript' and 'react', suggesting which updates and changes need to be done"
    },
    {
        "name":"digital_twins_backend",
        "prompt":"I need an AI Agent specialized in digital twins. I'll pass a specific problem and the theory about it. He must returns to me all the python code and structure to simulate this problem as backend"
    },
    {
        "name":"digital_twins_visual",
        "prompt":"I need an AI Agent specialized in digital twins. I'll pass a specific problem and the theory about it, as also the backend for the problem. He must returns a suggestion to create a visualization for this, showing the code for API's, the codes for builting amazing 3d visualizations, etc. He is specialized in any technology about visual computation, physics informed neural networks and 3d things."
    },
    {
        "name":"it_projects_distribution",
        "prompt":"I need an AI Agent that receives a task, the codes that were done and should create a repository project structure."
    },
    {
        "name":"automatic_pc",
        "prompt":"I need an AI Agent that receives a project structure and respective codes and shows the code in python to create the whole project"
    },
    {
        "name":"cybersecurity_analyst",
        "prompt":"I need an AI Agent specialized in cybersecurity. He must master red and blue tasks and show the user the way, guideline and codes to solve any problem"
    },
    {
        "name":"nasa_specialist",
        "prompt":"I need an AI Agent specialized in spatial field problems. I pass to him for example: How is the physics behind the rockets? And he should return to me all the info about it, including materials, proportions, way to build, codes, etc"
    },
    {
        "name":"scraper_specialist",
        "prompt":"I need an AI Agent specialized in web scraping. I should pass to him a task anb the html of page and he must returns me a way to collect the data. If i have any issues with the code he provided me, he must solve it correctly for me, as a specialist"
    },
    {
        "name":"infrastructure_specialist",
        "prompt":"I need an AI Agent specialized in IT infrastructure. He will receive a project and the whole code and he must built a solid, scalable and secure infrastructure for the project"
    }
]

for agent in list_of_agents:
    save_agent_structure(agent['prompt'], agent['name'])
    print(f"Agent {agent['name']} saved successfully")
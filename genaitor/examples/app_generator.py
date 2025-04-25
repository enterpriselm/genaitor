import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
from core import Orchestrator, Flow, ExecutionMode
from presets.agents import (
    dev_requirements_agent, backend_agent, frontend_agent, cicd_agent
)

import re

def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '_', filename)

def save_files_from_markdown(content: str, base_path="test_project"):
    os.makedirs(base_path, exist_ok=True)
    current_file = None
    file_lines = []

    for line in content.splitlines():
        if line.startswith("**") and line.endswith("**"):
            if current_file and file_lines:
                safe_filename = sanitize_filename(current_file)
                file_path = os.path.join(base_path, safe_filename)
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(file_lines).strip())
                file_lines = []

            current_file = line.strip("*").strip()

        elif current_file:
            file_lines.append(line)

    if current_file and file_lines:
        safe_filename = sanitize_filename(current_file)
        file_path = os.path.join(base_path, safe_filename)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(file_lines).strip())

def start_local_servers(base_path="generated_project"):
    backend_path = os.path.join(base_path, "backend")
    frontend_path = os.path.join(base_path, "frontend")

    print("\n🚀 Starting Backend (FastAPI)...")
    subprocess.Popen(["uvicorn", "main:app", "--reload"], cwd=backend_path)

    print("\n🚀 Starting Frontend (Vite)...")
    subprocess.Popen(["npm", "install"], cwd=frontend_path, shell=True).wait()
    subprocess.Popen(["npm", "run", "dev"], cwd=frontend_path, shell=True)

async def main(idea):
    print("\n🤖 Agentic SaaS Builder Initializing...\n")

    agents = {
        "requirements": dev_requirements_agent,
        "backend": backend_agent,
        "frontend": frontend_agent,
        "cicd": cicd_agent
    }

    orchestrator = Orchestrator(
        agents=agents,
        flows={
            "app_flow": Flow(
                agents=["requirements", "backend", "frontend", "cicd"],
                context_pass=[True, True, True, True]
            )
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    print(f"[App Idea]\n{idea}\n{'=' * 60}")

    result = await orchestrator.process_request(idea, flow_name='app_flow')

    if result["success"]:
        print("\n📁 Saving agents files...")

        for agent_key, agent_output in result["content"].items():
            print(f"🧠 Processing agent '{agent_key}'...")

            content = getattr(agent_output, "content", None)
            if isinstance(content, str):
                save_path = os.path.join("test_project", agent_key)
                save_files_from_markdown(content, base_path=save_path)
                print(f"✅ Files saved in '{save_path}/'")
            else:
                print(f"⚠️ Invalid or missing content in '{agent_key}'")

        print("\n📂 Project saved in 'generated_project/'\n")
        start_local_servers()

    else:
        print(f"❌ Error: {result['error']}")

if __name__ == "__main__":
    idea = input("Give an App idea.\n\n")
    asyncio.run(main(idea))
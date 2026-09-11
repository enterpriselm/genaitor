import asyncio
import os
import subprocess

from pinnaitor.core import Orchestrator, Flow, ExecutionMode
from pinnaitor.presets.agents import debugging_agent


class AutoDebuggingFlow:
    def __init__(self, agent_name="debugging_agent"):
        self.orchestrator = Orchestrator(
            agents={agent_name: debugging_agent},
            flows={
                "debugging_flow": Flow(agents=[agent_name], context_pass=[True])
            },
            mode=ExecutionMode.SEQUENTIAL
        )
        self.agent_name = agent_name

    async def run_debugging_loop(self, file_path: str):
        stderr_logs = "init"
        
        while stderr_logs:
            # Read and clean code
            with open(file_path, 'r') as f:
                code = f.read()
            cleaned_code = code.replace('```python', '').replace('```', '')

            with open(file_path, 'w') as f:
                f.write(cleaned_code)

            # Re-read cleaned code
            with open(file_path, 'r') as f:
                code = f.read()

            # Run script and capture logs
            try:
                result = subprocess.run(
                    ["python", file_path],
                    capture_output=True,
                    text=True
                )

                stdout_logs = result.stdout
                stderr_logs = result.stderr

                print("Standard Output:\n", stdout_logs)
                print("Error Logs:\n", stderr_logs)

            except Exception as e:
                print("Error running the script:", str(e))
                break

            # If errors exist, call debugging agent
            if stderr_logs:
                input_data = {f"code : {code}\nError: {stderr_logs}"}

                try:
                    result = await self.orchestrator.process_request(
                        input_data, flow_name="debugging_flow"
                    )

                    if result["success"]:
                        code = result['content'][self.agent_name].content.strip()
                        code = code.replace('```python', '').replace('```', '')
                        with open(file_path, 'w') as f:
                            f.write(code)
                    else:
                        print(f"\nError: {result['error']}")
                        break

                except Exception as e:
                    print(f"\nError: {str(e)}")
                    break


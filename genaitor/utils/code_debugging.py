import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.genaitor.core import Orchestrator, Flow, ExecutionMode
from src.genaitor.presets.agents import debugging_agent
import subprocess

async def main(file_path=r'examples\files\numerical_modeling.py'):
    orchestrator = Orchestrator(
        agents={"debugging_agent": debugging_agent},
        flows={
            "debugging_flow": Flow(agents=["debugging_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    while stderr_logs!='':
        with open(file_path, 'r') as f:
            code = f.read()

        cleaned_code = code.replace('```python','').replace('```','')

        with open(file_path, 'w') as f:
            f.write(cleaned_code)

        with open(file_path, 'r') as f:
            code = f.read()

        try:
            result = subprocess.run(
                ["python", file_path], 
                capture_output=True, text=True
            )
        
            stdout_logs = result.stdout
            stderr_logs = result.stderr

            print("Standard Output:\n", stdout_logs)
            print("Error Logs:\n", stderr_logs)

        except Exception as e:
            print("Error running the script:", str(e))    

        if stderr_logs!='':
            input_data = {f"code : {code}\nError: {stderr_logs}"}

            try:
                result = await orchestrator.process_request(input_data, flow_name='debugging_flow')
                if result["success"]:
                    code=result['content']['debugging_agent'].content.strip()
                    code = code.replace('```python','').replace('```','')
                    with open(file_path, 'w') as f:
                        f.write(code)                    
                else:
                    print(f"\nError: {result['error']}")
        
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
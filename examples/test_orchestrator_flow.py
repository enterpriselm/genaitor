import sys
import os

# Adiciona o diretório src ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import asyncio
from core import Agent, Task, Orchestrator, TaskResult, ExecutionMode
from llm import GeminiProvider, GeminiConfig

# Define a mock task for testing
class MockTask(Task):
    def __init__(self, name: str):
        super().__initf__(description=f"Mock task for {name}", goal="Test goal", output_format="Test output")
        self.name = name

    def execute(self, input_data: str) -> TaskResult:
        # Simulate processing and returning a result
        return TaskResult(success=True, content=f"Processed by {self.name}: {input_data}", metadata={})

async def main():
    # Create mock agents
    agent1 = Agent(role="Agent 1", tasks=[MockTask("Agent 1")], llm_provider=None)
    agent2 = Agent(role="Agent 2", tasks=[MockTask("Agent 2")], llm_provider=None)

    # Setup orchestrator with sequential processing
    orchestrator = Orchestrator(
        agents={"agent1": agent1, "agent2": agent2},
        mode=ExecutionMode.SEQUENTIAL
    )

    # Initial input for the first agent
    initial_input = "Initial data for processing."

    # Process the request
    result = await orchestrator.process_request(initial_input)

    # Check the results
    assert result["success"] is True, "The orchestrator should succeed."
    assert "agent1" in result["content"], "The result should contain agent1's output."
    assert "agent2" in result["content"], "The result should contain agent2's output."
    
    # Check the content of agent1
    agent1_output = result["content"]["agent1"]
    assert agent1_output["success"] is True, "Agent 1 should succeed."
    assert agent1_output["content"] == "Processed by Agent 1: Initial data for processing.", "Agent 1 output is incorrect."

    # Check the content of agent2
    agent2_output = result["content"]["agent2"]
    assert agent2_output["success"] is True, "Agent 2 should succeed."
    assert agent2_output["content"] == "Processed by Agent 2: Processed by Agent 1: Initial data for processing.", "Agent 2 output is incorrect."

    print("All tests passed!")

if __name__ == "__main__":
    asyncio.run(main()) 
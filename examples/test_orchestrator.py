import pytest
from src.core import Agent, Task, Orchestrator, TaskResult, ExecutionMode, Flow
from src.llm import GeminiProvider, GeminiConfig

# Mock Task class for testing
class MockTask(Task):
    def __init__(self, description: str, goal: str, output_format: str):
        super().__init__(description, goal, output_format)

    def execute(self, input_data: str) -> TaskResult:
        if input_data == "error":
            return TaskResult(success=False, content=None, error="Mock error")
        return TaskResult(success=True, content=f"Processed: {input_data}")

# Test cases for Orchestrator
def test_orchestrator_sequential_processing():
    task1 = MockTask("Task 1", "Goal 1", "Output format 1")
    task2 = MockTask("Task 2", "Goal 2", "Output format 2")
    
    agent1 = Agent(role="Agent 1", tasks=[task1])
    agent2 = Agent(role="Agent 2", tasks=[task2])
    
    orchestrator = Orchestrator(
        agents={"agent1": agent1, "agent2": agent2},
        flows={"default": Flow(agents=["agent1", "agent2"], context_pass=[True, True])},
        mode=ExecutionMode.SEQUENTIAL
    )
    
    result = orchestrator.process_request("Test input", "default")
    
    assert result["success"] is True
    assert result["content"]["agent1"].success is True
    assert result["content"]["agent2"].success is True
    assert result["content"]["agent1"].content == "Processed: Test input"
    assert result["content"]["agent2"].content == "Processed: Test input"

def test_orchestrator_error_handling():
    task1 = MockTask("Task 1", "Goal 1", "Output format 1")
    task2 = MockTask("Task 2", "Goal 2", "Output format 2")
    
    agent1 = Agent(role="Agent 1", tasks=[task1])
    agent2 = Agent(role="Agent 2", tasks=[task2])
    
    orchestrator = Orchestrator(
        agents={"agent1": agent1, "agent2": agent2},
        flows={"default": Flow(agents=["agent1", "agent2"], context_pass=[True, True])},
        mode=ExecutionMode.SEQUENTIAL
    )
    
    result = orchestrator.process_request("error", "default")
    
    assert result["success"] is False
    assert "error" in result
    assert result["error"] == "An error occurred while processing the request: Mock error"

# Test cases for Agent
def test_agent_task_execution():
    task = MockTask("Test Task", "Test Goal", "Test Output")
    agent = Agent(role="Test Agent", tasks=[task])
    
    result = agent.execute_task(task, "Test input")
    
    assert result.success is True
    assert result.content == "Processed: Test input"

def test_agent_task_execution_error():
    task = MockTask("Error Task", "Test Goal", "Test Output")
    agent = Agent(role="Test Agent", tasks=[task])
    
    result = agent.execute_task(task, "error")
    
    assert result.success is False
    assert result.error == "Error executing task 'Error Task': Mock error"

if __name__ == "__main__":
    pytest.main()
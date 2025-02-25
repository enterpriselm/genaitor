import asyncio
import unittest
from src.core.agent import Agent
from src.core.base import Task, TaskResult, AgentRole

class MockTask(Task):
    def execute(self, input_data):
        return TaskResult(success=True, content="Task executed", error=None)

class TestAgent(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(role=AgentRole.MAIN, tasks=[MockTask(description="Mock Task", goal="Test", output_format="JSON")])

    def test_execute_task_success(self):
        result = self.agent.execute_task(self.agent.tasks[0], {})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Task executed")

    async def test_process_request(self):
        result = await self.agent.process_request({}, {})
        print(result)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Task executed")

if __name__ == '__main__':
    unittest.main() 
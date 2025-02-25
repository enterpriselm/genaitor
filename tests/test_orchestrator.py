import asyncio
import unittest
from src.core.orchestrator import Orchestrator, Flow
from src.core.agent import Agent
from src.core.base import TaskResult

class MockAgent(Agent):
    async def process_request(self, request, context):
        return TaskResult(success=True, content="Processed", error=None)

class TestOrchestrator(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent(role="test", tasks=[])
        self.orchestrator = Orchestrator(agents={"test_agent": self.agent}, flows={"test_flow": Flow(agents=["test_agent"], context_pass=[True])})

    async def test_process_request(self):
        result = await self.orchestrator.process_request({}, "test_flow")
        self.assertTrue(result["success"])
        self.assertEqual(result["content"]["test_agent"], "Processed")

if __name__ == '__main__':
    unittest.main() 
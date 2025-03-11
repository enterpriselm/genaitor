import asyncio
import unittest
from llm.fine_tuning import FineTuningTask
from src.core.base import TaskResult

class MockLLMProvider:
    def generate(self, prompt):
        return {"learning_rate": 0.0001, "batch_size": 8, "warmup_steps": 100}

class TestFineTuningTask(unittest.TestCase):
    def setUp(self):
        self.task = FineTuningTask("distilgpt2", "wikitext", "./trained_model", MockLLMProvider())

    async def test_execute(self):
        result = await self.task.execute()
        print("DEBUG", result)
        self.assertTrue(result.success)
        self.assertEqual(result.content, "./trained_model")

if __name__ == '__main__':
    unittest.main() 
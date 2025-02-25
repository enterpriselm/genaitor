import unittest
from src.core.base import Task, TaskResult

class MockTask(Task):
    def execute(self, input_data):
        return TaskResult(success=True, content="Executed", error=None)

class TestTask(unittest.TestCase):
    def setUp(self):
        self.task = MockTask(description="Test Task", goal="Test", output_format="JSON")

    def test_execute(self):
        result = self.task.execute({})
        self.assertTrue(result.success)
        self.assertEqual(result.content, "Executed")

if __name__ == '__main__':
    unittest.main() 
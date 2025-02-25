import unittest
from src.llm.base import LLMProvider, LLMConfig

class MockLLMProvider(LLMProvider):
    def stream(self, *args, **kwargs):
        pass
    def generate(self, prompt: str, **kwargs) -> str:
        return "Generated text"

class TestLLMProvider(unittest.TestCase):
    def setUp(self):
        self.provider = MockLLMProvider(LLMConfig())

    def test_generate(self):
        result = self.provider.generate("Test prompt")
        self.assertEqual(result, "Generated text")

if __name__ == '__main__':
    unittest.main() 
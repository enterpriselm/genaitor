import unittest
from genaitor.utils.text_splitter import TextSplitter

class TestTextSplitter(unittest.TestCase):
    def setUp(self):
        self.splitter = TextSplitter(chunk_size=10, chunk_overlap=2)

    def test_split_text(self):
        text = "This is a test text for splitting."
        chunks = self.splitter.split_text(text)
        self.assertGreater(len(chunks), 1)

if __name__ == '__main__':
    unittest.main() 
# Genaitor

Genaitor is a framework for building AI agents that can perform various tasks using generative models.

## Installation

To install the required dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/genaitor.git
   cd genaitor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Example

Here’s a simple example of how to create an agent that answers questions using a generative model:

```python
from src.core import Agent, Task
from src.llm import GeminiProvider, GeminiConfig

# Define a custom task
class QuestionAnsweringTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str):
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        Question: {input_data}
        Please provide a response following the format:
        {self.output_format}
        """
        return self.llm.generate(prompt)

# Configure the LLM provider
llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

# Create an agent
agent = Agent(name="QA Agent", task=QuestionAnsweringTask("Answering questions", "Provide accurate answers", "Text format", llm_provider))

# Execute a task
result = agent.task.execute("What is AI?")
print(result)
```

## Contribution Guidelines

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or suggestions, feel free to open an issue or contact the maintainers at [your-email@example.com].
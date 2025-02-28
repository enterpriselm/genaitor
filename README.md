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

## Diagram

![System Pipeline](https://github.com/enterpriselm/genaitor/blob/main/genaitor.jpg?raw=true)

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

### Multi-Agent Example

Here’s a simple example of how to create a flow using multiple agents:

```python
import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Define a base task (you could use different tasks for each agent)
class LLMTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Input: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": self.description}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

# Configure the LLM provider
llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

# Generating two specific tasks
qa_task = LLMTask(
    description="Question Answering",
    goal="Provide clear and accurate responses",
    output_format="Concise and informative",
    llm_provider=llm_provider
)
    
summarization_task = LLMTask(
    description="Text Summarization",
    goal="Summarize lengthy content into key points",
    output_format="Bullet points or short paragraph",
    llm_provider=llm_provider
)

# Create agents
qa_agent = Agent(
    role=AgentRole.SPECIALIST,
    tasks=[qa_task],
    llm_provider=llm_provider
)
summarization_agent = Agent(
    role=AgentRole.SUMMARIZER,
    tasks=[summarization_task],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={"qa_agent": qa_agent, "summarization_agent": summarization_agent},
    flows={
        "default_flow": Flow(agents=["qa_agent", "summarization_agent"], context_pass=[True,True])
    },
    mode=ExecutionMode.SEQUENTIAL
)
    
result_process = orchestrator.process_request('What is the impact of AI on modern healthcare?', flow_name='default_flow')
result = asyncio.run(result_process)
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

For any questions or suggestions, feel free to open an issue or contact the maintainers at executive.enterpriselm@gmail.com

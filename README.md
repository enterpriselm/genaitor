<p align="center">
  <img src="logo.png" alt="Genaitor Logo" width="300"/>
</p>

# GenAItor

A platform for AI Agents and AI Agents products generation.

## Overview

GenAItor is a cutting-edge platform designed to generate AI agents and related products that help automate complex tasks and processes. It leverages state-of-the-art machine learning libraries and tools to deliver flexible and scalable AI solutions.

To install the required dependencies, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/enterpriselm/genaitor.git
   cd genaitor
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -e .
   ```

4. Add API_KEY for llm (Gemini set as main, but you can use Anthropic, OpenAI, DeepSeek, Grok, Ollama or a custom LLM model):
   ```bash
   echo "API_KEY=your_gemini_api_key" >> .env
   ```

## Features

- Generate AI agents for a variety of use cases.
- Modular architecture with components such as `core`, `llm`, `utils`, and `presets`.
- Support for multiple data processing and communication protocols.
- Integration with popular libraries like Transformers, Langchain, and more.

## Usage

### Basic Example

Here’s a simple example of how to create an agent that answers questions using a generative model:

```python
from genaitor.core import Agent, Task
from genaitor.llm import GeminiProvider, GeminiConfig

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
from genaitor.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from genaitor.llm import GeminiProvider, GeminiConfig

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

## Examples usage

Here is a simple guideline for running the examples

### Streamlit APPs

```bash

streamlit run genaitor\apps\pinneaple.py
```

### General examples

```bash

python genaitor\examples\autism_assistant.py
```

## Demo Videos

Here are some demo videos showcasing Genaitor in action:

- [Apps Generation](https://youtu.be/aJboXG3RvsA)
- [OCR and Power Apps Automatization](https://youtu.be/VvIb7x3PJWQ)
- [Satellite Images Analysis](https://youtu.be/hsjanmnCxJ4)
- [PINNeAPPle](https://youtu.be/AbYr3F_v5OA)

## FAQ

Why should I use this framework over others like LangChain, LangGraph, CrewAI or LlamaIndex?

While popular frameworks like LangChain, LangGraph, and LlamaIndex are powerful, they are primarily designed as general-purpose agentic frameworks.
Our framework is specifically optimized for Scientific Machine Learning (SciML) applications and offers the following key advantages:

- Specific focus on Scientific Machine Learning:
Unlike generalist frameworks, we prioritize workflows tailored for scientific and physics-based AI tasks, where agent behavior often requires structured reasoning and domain-specific knowledge handling.

- Greater control and transparency:
Our design provides developers with direct access to agent modeling and lifecycle management.
You are not tied to predefined abstractions or "black-box" architectures, allowing full customization to match scientific workflows.

- Reduced learning curve:
Our framework minimizes unnecessary complexity.
Users can build efficient agents with a much simpler and more intuitive interface, without needing to dive deep into multiple layers of abstractions before achieving results.


Is this framework compatible with LangChain or LlamaIndex?

Our framework is independent but compatible with most libraries from the ecosystem.
You can integrate components like LlamaIndex for document retrieval or LangChain tools if needed, while still maintaining full control over the agent lifecycle inside our framework.

What kind of Scientific Machine Learning tasks is this framework suited for?

This framework is designed for tasks such as:

- Physics-informed problem solving

- Scientific reasoning and simulation control

- AI-driven research assistants for scientific domains

- Autonomous agents for data-driven discovery processes

- Interaction with physical simulation APIs, datasets, and analytical tools


If your use case involves structured reasoning, scientific models, or physics-based tasks, this framework provides the flexibility and precision you need.

## Contribution Guidelines

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes and commit (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Create a pull request.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, feel free to open an issue or contact the maintainers at executive.enterpriselm@gmail.com or the main author Yan Barros at https://www.linkedin.com/in/yan-barros-yan

You can also check our landing-page to more news:

enterpriselm.github.io/home

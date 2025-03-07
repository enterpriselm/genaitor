import asyncio
from src.core import Agent, Task, Orchestrator, Flow, ExecutionMode, AgentRole, TaskResult
from src.llm import GeminiProvider, GeminiConfig
from src.utils.media_processor import extract_text_from_pdf

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

class SummarizationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Scientific Paper Summarization",
            goal="Summarize key points of a scientific paper",
            output_format="Bullet points highlighting main findings"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input (Scientific Paper): {input_data}

        Please provide a response following the format:
        {self.output_format}
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

class LinkedinPostTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="LinkedIn Post Generation",
            goal="Create an engaging LinkedIn post based on a scientific paper summary",
            output_format="A LinkedIn-friendly post with hashtags and a call to action"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input (Scientific Paper Summary): {input_data}

        Please provide a response following the format:
        {self.output_format}
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))

summarization_agent = Agent(
    role=AgentRole.SUMMARIZER,
    tasks=[SummarizationTask(llm_provider)],
    llm_provider=llm_provider
)

post_generation_agent = Agent(
    role=AgentRole.CUSTOM,
    tasks=[LinkedinPostTask(llm_provider)],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={
        "summarization_agent": summarization_agent,
        "post_generation_agent": post_generation_agent
    },
    flows={
        "linkedin_post_flow": Flow(
            agents=["summarization_agent", "post_generation_agent"],
            context_pass=[True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

async def run_flow(scientific_paper_text):
    result_process = orchestrator.process_request(scientific_paper_text, flow_name="linkedin_post_flow")
    result = await result_process
    print(result)

paper_path = ""
paper_text = extract_text_from_pdf(paper_path)
asyncio.run(run_flow(paper_text))

import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

class AudienceResearchTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Audience Research",
            "Analyze target audience and suggest the best email tone and approach",
            "JSON format with audience insights and tone suggestions"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Campaign Details: {input_data}

        Please analyze the audience and provide insights such as demographics, interests, and the best email tone (e.g., formal, casual, persuasive).
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "audience_research"})

class EmailContentGenerationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Email Content Generation",
            "Generate an email draft based on the campaign and audience",
            "Structured email content including subject line, body, and CTA"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Audience Insights: {input_data}

        Generate an engaging email including:
        - Subject Line
        - Opening Hook
        - Main Message
        - Call to Action (CTA)
        - Closing Statement
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "email_generation"})

class EmailOptimizationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Email Optimization",
            "Refine email content for clarity, engagement, and conversion",
            "Optimized email content with persuasive elements"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Draft Email Content: {input_data}

        Improve the email copy by enhancing readability, engagement, and conversion effectiveness. Ensure the subject line is compelling and the CTA is action-driven.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "email_optimization"})

class EmailPersonalizationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Email Personalization",
            "Adapt email content for different audience segments",
            "Different versions of the email for specific audience segments"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Optimized Email Content: {input_data}

        Create different variations of the email for different audience segments (e.g., new customers, loyal customers, inactive users).
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "email_personalization"})

audience_research_agent = Agent(
    role=AgentRole.ANALYST,
    tasks=[AudienceResearchTask(llm_provider)],
    llm_provider=llm_provider
)

email_generation_agent = Agent(
    role=AgentRole.CONTENT_CREATOR,
    tasks=[EmailContentGenerationTask(llm_provider)],
    llm_provider=llm_provider
)

email_optimization_agent = Agent(
    role=AgentRole.COPYWRITER,
    tasks=[EmailOptimizationTask(llm_provider)],
    llm_provider=llm_provider
)

email_personalization_agent = Agent(
    role=AgentRole.MARKETER,
    tasks=[EmailPersonalizationTask(llm_provider)],
    llm_provider=llm_provider
)

orchestrator = Orchestrator(
    agents={
        "audience_research_agent": audience_research_agent,
        "email_generation_agent": email_generation_agent,
        "email_optimization_agent": email_optimization_agent,
        "email_personalization_agent": email_personalization_agent
    },
    flows={
        "email_marketing_flow": Flow(
            agents=["audience_research_agent", "email_generation_agent", "email_optimization_agent", "email_personalization_agent"],
            context_pass=[True, True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

result_process = orchestrator.process_request(
    "Email campaign for a new product launch targeting tech enthusiasts",
    flow_name="email_marketing_flow"
)
result = asyncio.run(result_process)

print(result)

import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Configuração do LLM Provider
llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

# 1. Análise de Documentos (PDF/PPT)
class DocumentAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Document Analysis",
            "Extract relevant content from PDF or PPT documents",
            "JSON format with extracted content"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Document: {input_data}

        Extract the relevant content from the document. Provide an overview or key points from the document in a structured format (e.g., bullet points or paragraphs).
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "document_analysis"})

# 2. Análise de Pergunta
class QuestionAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Question Analysis",
            "Interpret the user's question and prepare it for content search",
            "JSON format with the question and intended answer structure"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Question: {input_data}

        Analyze the question and identify key terms or concepts that should be looked for in the document content.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "question_analysis"})

# 3. Busca por Resposta
class AnswerSearchTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Answer Search",
            "Search for the answer within the extracted content from the document",
            "JSON format with the search result"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Document Content: {input_data['document_content']}
        Question Analysis: {input_data['question_analysis']}

        Search for the answer to the question in the document content and return the most relevant sections or text.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "answer_search"})

# 4. Geração de Resposta
class ResponseGenerationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Response Generation",
            "Generate a final response based on the found answer and provide it in a clear format",
            "Final answer in text format"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Answer Search Result: {input_data['answer_search_result']}

        Generate a clear and concise response to the user, summarizing the answer found in the document. Ensure it is relevant and understandable.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "response_generation"})

# Criando Agentes
document_analysis_agent = Agent(
    role=AgentRole.DOCUMENT_ANALYST,
    tasks=[DocumentAnalysisTask(llm_provider)],
    llm_provider=llm_provider
)

question_analysis_agent = Agent(
    role=AgentRole.QUESTION_ANALYST,
    tasks=[QuestionAnalysisTask(llm_provider)],
    llm_provider=llm_provider
)

answer_search_agent = Agent(
    role=AgentRole.SEARCH_AGENT,
    tasks=[AnswerSearchTask(llm_provider)],
    llm_provider=llm_provider
)

response_generation_agent = Agent(
    role=AgentRole.RESPONDER,
    tasks=[ResponseGenerationTask(llm_provider)],
    llm_provider=llm_provider
)

# Criando o Orquestrador
orchestrator = Orchestrator(
    agents={
        "document_analysis_agent": document_analysis_agent,
        "question_analysis_agent": question_analysis_agent,
        "answer_search_agent": answer_search_agent,
        "response_generation_agent": response_generation_agent
    },
    flows={
        "pdf_ppt_qna_flow": Flow(
            agents=["document_analysis_agent", "question_analysis_agent", "answer_search_agent", "response_generation_agent"],
            context_pass=[True, True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

# Executando o fluxo
result_process = orchestrator.process_request(
    {"document_content": "Content of PDF or PPT", "question": "What is the main idea of this document?"},
    flow_name="pdf_ppt_qna_flow"
)
result = asyncio.run(result_process)

print(result)

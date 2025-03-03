import requests
from bs4 import BeautifulSoup
import csv
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Configuração do LLM Provider (caso necessário para análise adicional)
llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

# 1. Coleta de Dados da Web
class WebScrapingTask(Task):
    def __init__(self):
        super().__init__(
            "Web Scraping",
            "Perform web scraping on the specified URL to extract required data",
            "HTML content or JSON with extracted data"
        )

    def execute(self, url: str) -> TaskResult:
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # Suponhamos que queremos extrair todos os títulos de um site
            data = [title.get_text() for title in soup.find_all('h1')]
            return TaskResult(success=True, content=data, metadata={"task": "web_scraping"})
        except Exception as e:
            return TaskResult(success=False, content=str(e), metadata={"task": "web_scraping"})

# 2. Processamento de Dados
class DataProcessingTask(Task):
    def __init__(self):
        super().__init__(
            "Data Processing",
            "Clean and structure the scraped data for further use",
            "Structured data ready for storage"
        )

    def execute(self, raw_data: str) -> TaskResult:
        # Exemplo simples de processamento, por exemplo, remoção de espaços extras
        processed_data = [item.strip() for item in raw_data if item.strip()]
        return TaskResult(success=True, content=processed_data, metadata={"task": "data_processing"})

# 3. Armazenamento de Dados
class DataStorageTask(Task):
    def __init__(self):
        super().__init__(
            "Data Storage",
            "Store the processed data in a suitable format (e.g., CSV, database)",
            "Storage confirmation"
        )

    def execute(self, processed_data: list) -> TaskResult:
        try:
            with open('scraped_data.csv', 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['Title'])
                for data in processed_data:
                    writer.writerow([data])
            return TaskResult(success=True, content="Data successfully stored in CSV", metadata={"task": "data_storage"})
        except Exception as e:
            return TaskResult(success=False, content=str(e), metadata={"task": "data_storage"})

# Criando Agentes
web_scraping_agent = Agent(
    role=AgentRole.DATA_COLLECTOR,
    tasks=[WebScrapingTask()],
    llm_provider=llm_provider
)

data_processing_agent = Agent(
    role=AgentRole.DATA_PROCESSOR,
    tasks=[DataProcessingTask()],
    llm_provider=llm_provider
)

data_storage_agent = Agent(
    role=AgentRole.DATA_STORAGE,
    tasks=[DataStorageTask()],
    llm_provider=llm_provider
)

# Criando o Orquestrador
orchestrator = Orchestrator(
    agents={
        "web_scraping_agent": web_scraping_agent,
        "data_processing_agent": data_processing_agent,
        "data_storage_agent": data_storage_agent
    },
    flows={
        "web_scraping_flow": Flow(
            agents=["web_scraping_agent", "data_processing_agent", "data_storage_agent"],
            context_pass=[True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

# Executando o fluxo
result_process = orchestrator.process_request(
    {"url": "https://example.com"},  # URL que deseja realizar o scraping
    flow_name="web_scraping_flow"
)
result = asyncio.run(result_process)

print(result)

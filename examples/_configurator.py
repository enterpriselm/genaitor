import asyncio
from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig

# Configuração do LLM Provider
llm_provider = GeminiProvider(GeminiConfig(api_key="your_api_key"))

# 1. Análise de Preferências
class PreferencesAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Preferences Analysis",
            "Analyze the customer preferences and create a list of viable car models and options",
            "JSON format with the analyzed preferences"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Customer Preferences: {input_data}

        Analyze the customer preferences and provide a list of available options:
        - Car model
        - Color preference
        - Additional accessories
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "preferences_analysis"})

# 2. Cálculo de Pagamento
class PaymentCalculationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Payment Calculation",
            "Calculate the payment conditions based on financing options and customer budget",
            "JSON format with payment details and final amount"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Customer Information: {input_data}

        Calculate the following:
        - Final price considering the car model, accessories, and color
        - Financing conditions (20% down payment, 24 months)
        - Ensure the total amount fits within the customer’s budget of R$ 80,000.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "payment_calculation"})

# 3. Criação de Proposta
class ProposalGenerationTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Proposal Generation",
            "Generate a personalized proposal with the final price, payment options, and accessories included",
            "Detailed proposal with car model, payment terms, accessories, and total cost"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Customer Preferences and Payment Details: {input_data}

        Generate a proposal with:
        - Car model and accessories
        - Payment conditions (financing, entry, number of installments)
        - Final total cost
        Ensure the proposal is clear and structured.
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "proposal_generation"})

# 4. Revisão de Proposta
class ProposalReviewTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            "Proposal Review",
            "Review the proposal to ensure it is clear, concise, and covers all customer needs",
            "Clear and final version of the proposal"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Generated Proposal: {input_data}

        Review the proposal to ensure:
        - It is clear and understandable
        - All customer preferences are included
        - Payment terms are clearly stated
        - The proposal fits within the budget
        """
        response = self.llm.generate(prompt)
        return TaskResult(success=True, content=response, metadata={"task": "proposal_review"})

# Criando Agentes
preferences_analysis_agent = Agent(
    role=AgentRole.ANALYST,
    tasks=[PreferencesAnalysisTask(llm_provider)],
    llm_provider=llm_provider
)

payment_calculation_agent = Agent(
    role=AgentRole.FINANCIAL_ANALYST,
    tasks=[PaymentCalculationTask(llm_provider)],
    llm_provider=llm_provider
)

proposal_generation_agent = Agent(
    role=AgentRole.PROPOSAL_CREATOR,
    tasks=[ProposalGenerationTask(llm_provider)],
    llm_provider=llm_provider
)

proposal_review_agent = Agent(
    role=AgentRole.REVIEWER,
    tasks=[ProposalReviewTask(llm_provider)],
    llm_provider=llm_provider
)

# Criando o Orquestrador
orchestrator = Orchestrator(
    agents={
        "preferences_analysis_agent": preferences_analysis_agent,
        "payment_calculation_agent": payment_calculation_agent,
        "proposal_generation_agent": proposal_generation_agent,
        "proposal_review_agent": proposal_review_agent
    },
    flows={
        "car_purchase_proposal_flow": Flow(
            agents=["preferences_analysis_agent", "payment_calculation_agent", "proposal_generation_agent", "proposal_review_agent"],
            context_pass=[True, True, True, True]
        )
    },
    mode=ExecutionMode.SEQUENTIAL
)

# Executando o fluxo
result_process = orchestrator.process_request(
    "Customer wants to purchase a Sedan with black color, financing with 20% down payment and 24 months installment, includes premium sound system, leather seats, and rearview camera, with a budget of R$ 80,000.",
    flow_name="car_purchase_proposal_flow"
)
result = asyncio.run(result_process)

print(result)

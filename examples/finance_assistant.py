import asyncio
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
from dotenv import load_dotenv
load_dotenv('.env')

# Define custom task for suggesting investment strategies based on market data
class InvestmentStrategyTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the following financial market data and historical trends, suggest strategies that minimize risk and maximize returns in a volatile market:
        
        Input: {input_data}
        
        Focus on diversification, risk management, and optimization for high volatility.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "investment_strategy"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


# Define a task to predict the probability of default and suggest improvements in credit granting
class CreditRiskPredictionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the credit data of clients and their transaction history, predict the probability of default and suggest improvements in the credit granting process:
        
        Input: {input_data}
        
        Focus on identifying potential risks and improving decision-making in credit allocation.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "credit_risk_prediction"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


# Define a task to adjust portfolio allocations based on performance metrics
class PortfolioOptimizationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the performance metrics of a diversified investment portfolio, suggest adjustments to optimize the expected return within a desired level of risk:
        
        Input: {input_data}
        
        Focus on asset allocation strategies and balancing risk versus return.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "portfolio_optimization"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


# Define a task to detect fraudulent transactions based on patterns
class FraudDetectionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the transaction history of clients, detect anomalous patterns that could indicate fraudulent activities such as money laundering or unauthorized transactions:
        
        Input: {input_data}
        
        Focus on identifying outliers, irregular patterns, and suspicious behavior in financial data.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "fraud_detection"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


async def main():
    print("\nInitializing Financial Analysis System...")
    test_keys = [os.getenv('API_KEY_1'), os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)

    # Instantiate tasks
    investment_task = InvestmentStrategyTask(
        description="Analyze financial market data and suggest investment strategies",
        goal="Minimize risk and maximize returns in high volatility",
        output_format="Recommended strategies",
        llm_provider=provider
    )

    credit_risk_task = CreditRiskPredictionTask(
        description="Predict credit risk and suggest improvements in credit granting",
        goal="Improve credit decision-making and minimize defaults",
        output_format="Risk prediction and suggestions for improvement",
        llm_provider=provider
    )

    portfolio_task = PortfolioOptimizationTask(
        description="Optimize portfolio allocations based on performance metrics",
        goal="Maximize return within a given risk level",
        output_format="Recommended portfolio adjustments",
        llm_provider=provider
    )

    fraud_detection_task = FraudDetectionTask(
        description="Detect anomalous patterns in financial transactions",
        goal="Identify fraudulent activities like money laundering or unauthorized transactions",
        output_format="Detected fraud patterns",
        llm_provider=provider
    )

    # Create an agent for financial analysis
    financial_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[investment_task, credit_risk_task, portfolio_task, fraud_detection_task],
        llm_provider=provider
    )
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"financial_agent": financial_agent},
        flows={
            "default_flow": Flow(agents=["financial_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Market Data: S&P 500, DXY, crude oil prices. Suggest strategies to minimize risk and maximize returns in high volatility.",
        "Customer Credit Data: Transaction history of clients with 60% probability of default. Predict and suggest improvements in credit granting process.",
        "Portfolio Data: Diverse portfolio of stocks, bonds, and real estate. Suggest adjustments to optimize risk-return balance.",
        "Transaction Data: Client transactions with unusual high-frequency trading. Detect potential fraudulent activities."
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("financial_agent")
                    if content and content.success:
                        print("\nSuggested Actions:")
                        print("-" * 80)
                        
                        formatted_text = content.content.strip()
                        
                        formatted_text = formatted_text.replace("**", "")
                        
                        for line in formatted_text.split('\n'):
                            if line.strip():
                                print(line)
                            else:
                                print()
                    else:
                        print("Empty response received")
                else:
                    print(result["content"] or "Empty response")
            else:
                print(f"\nError: {result['error']}")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            break

if __name__ == "__main__":
    asyncio.run(main())

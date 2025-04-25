from typing import Any, Dict
import json

from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from fastapi import FastAPI

from core import Task, TaskResult
from presets.providers import gemini_provider

provider = gemini_provider()


class QuestionAnsweringTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Question: {input_data}
        
        Please provide a response following the format:
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "qa"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class AgentCreationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the following input, create a new task that an agent can perform:
        
        Input: {input_data}
        
        Please describe the new task, including:
        1. Description of the task
        2. The goal of the task
        3. Output format
        
        Format it in a way that it can be understood and performed by an agent.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "agent_creation"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class AnomaliesDetection(Task):
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


class GeneralTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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


class AutismSupportTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
        Please provide a response following the format:
        {self.output_format}

        Uses the Hyperfocus to improve the answer or the learning path for the answer
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "autism_support"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class ProblemAnalysis(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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


class PINNModeling(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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


class CarPurchaseTask(Task):
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


class CVTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: Dict[str, Any]) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input:
        {json.dumps(input_data, indent=4)}

        Provide the response in the following format:
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


class ModelSelectionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following task and dataset characteristics, suggest the most suitable model type (ML or DL):
        
        Input: {input_data}
        
        Focus on model suitability, data type, and complexity of the task.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "model_selection"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class HyperparameterTuningTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following model and dataset details, suggest the best hyperparameter tuning methods and optimal values:
        
        Input: {input_data}
        
        Focus on tuning techniques like Grid Search, Random Search, Bayesian Optimization, and their impact on model performance.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "hyperparameter_tuning"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class ModelEvaluationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following model performance metrics, evaluate the model's strengths and weaknesses:
        
        Input: {input_data}
        
        Focus on precision, recall, F1-score, AUC-ROC, and other relevant evaluation metrics.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "model_evaluation"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class RegularizationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Given the following model details, suggest the most appropriate regularization techniques to avoid overfitting:
        
        Input: {input_data}
        
        Focus on techniques like L1/L2 regularization, dropout, early stopping, etc.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "regularization"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class StudentPerformanceAnalysisTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the following student performance data, identify gaps in understanding and suggest topics to focus on for improvement:
        
        Input: {input_data}
        
        Analyze the student’s performance, identify weak areas, and recommend topics that should be covered to improve understanding.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "student_performance_analysis"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class FutureDifficultiesPredictionTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the student’s past performance, predict which upcoming subjects may pose difficulties. Suggest adjustments to the teaching plan to address these potential challenges:
        
        Input: {input_data}
        
        Consider the student’s past struggles, their learning style, and the expected difficulty of future topics.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "future_difficulties_prediction"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class MaterialRecommendationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the student’s current knowledge and performance, suggest relevant materials and topics that could improve their understanding of the subject:
        
        Input: {input_data}
        
        Recommend learning materials (e.g., books, articles, online resources) and specific topics to focus on.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "material_recommendation"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class LanguageLearningActivityTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the student’s performance in language learning, suggest specific activities or practices to help improve their understanding of difficult areas:
        
        Input: {input_data}
        
        Recommend exercises, activities, or practical steps to address weaknesses identified in the language learning process.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "language_learning_activities"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class EmailTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: Dict[str, Any]) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input:
        {json.dumps(input_data, indent=4)}

        Provide the response in the following format:
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


class SummarizationTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: Dict[str, Any]) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input:
        {json.dumps(input_data, indent=4)}

        Provide the response in the following format:
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


class PinnHyperparameterTuningTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Based on the following input, suggest possible adjustments or modifications to the hyperparameters for better performance:
        
        Input: {input_data}
        
        Please review the architecture and training parameters and suggest any adjustments in the following areas:
        1. Learning rate
        2. Batch size
        3. Number of epochs
        4. Network architecture adjustments (e.g., number of layers, neurons per layer)
        5. Other relevant hyperparameters
        
        Format your response in a clear and actionable way, detailing any suggested changes.
        """

        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "pinn_tuning"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )


class WebScraping(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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


class UnityCodeGenerator(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        {input_data}
        
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
            return TaskResult(success=False, content=None, error=str(e))


class TravelTask(Task):
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


class TemporalSeriesForecasting(Task):
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


class AnalysisTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: Dict[str, Any]) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        {json.dumps(input_data, indent=4)}

        Provide the response in the following format:
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


class MatchAnalysisTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        {input_data}

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


class SecurityTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        {input_data}

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


class DisasterAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Environmental Disaster Analysis",
            goal="Detect environmental disasters such as floods, wildfires, or landslides using spectral bands",
            output_format="Detailed report of potential environmental disasters detected in the selected bands"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a detailed report on potential environmental disasters detected, such as floods, wildfires, or landslides.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class AgroAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Agricultural Analysis",
            goal="Monitor crop health and detect signs of water stress or pests using spectral bands",
            output_format="Crop health report, highlighting signs of water stress or pest infestation"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on crop health, highlighting signs of water stress or pest infestation.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class EcologicalAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Ecological Analysis",
            goal="Study local vegetation and ecosystems to detect signs of environmental degradation",
            output_format="Report on signs of environmental degradation in local ecosystems"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on signs of environmental degradation in local ecosystems.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class AirQualityAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Air Quality Analysis",
            goal="Detect air pollution, such as smoke or other contaminants in the spectral bands",
            output_format="Report on areas with detected air pollution"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on areas with detected air pollution, such as smoke or other contaminants.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class VegetationAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Vegetation Analysis",
            goal="Detect deforestation or changes in vegetation using spectral bands",
            output_format="Report on deforestation or changes in vegetation detected"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on deforestation or changes in vegetation detected.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class SoilMoistureAnalysisTask(Task):
    def __init__(self, llm_provider):
        super().__init__(
            description="Soil Moisture Analysis",
            goal="Study soil moisture and identify areas prone to drought or excess water",
            output_format="Report on soil moisture and regions prone to drought or excess water"
        )
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}

        Input data (spectral bands): {input_data}

        Please provide a report on soil moisture and areas prone to drought or excess water.
        """
        try:
            response = self.llm.generate(prompt)
            return TaskResult(success=True, content=response)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class FineTuningTask(Task):
    def __init__(self, model_name, dataset_name, output_dir, llm_provider):
        super().__init__(
            description="Fine-tuning strategy",
            goal="Train an LLM and suggest best hyperparameters",
            output_format="JSON format with training settings"
        )
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.llm = llm_provider

    def execute(self) -> TaskResult:
        try:
            prompt = f"""
            You are an expert in training language models.
            The model {self.model_name} will be fine-tuned using the dataset {self.dataset_name}.

            Requirements:

            Suggest the best hyperparameters (learning rate, batch size, warmup steps).
            Indicate if LoRA or QLoRA should be used to optimize memory.
            Report any potential risks or issues based on the dataset.
            Return a JSON in the format:

            {
                "learning_rate": 0.0001,
                "batch_size": 8,
                "warmup_steps": 100,
                "use_LoRA": true,
                "recommendations": "Avoid overfitting by using early stopping."
            }
            """
            hyperparams = self.llm.generate(prompt)
            
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            dataset = load_dataset(self.dataset_name)

            def tokenize_function(examples):
                return tokenizer(examples["text"], padding="max_length", truncation=True)
            
            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            
            training_args = TrainingArguments(
                output_dir=self.output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                per_device_train_batch_size=hyperparams.get("batch_size", 4),
                num_train_epochs=1,
                weight_decay=0.01
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["test"]
            )
            
            trainer.train()
            model.save_pretrained(self.output_dir)
            tokenizer.save_pretrained(self.output_dir)
            
            return TaskResult(success=True, content=self.output_dir)
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


class ModelDeploymentTask(Task):
    def __init__(self, model_dir, llm_provider):
        super().__init__(
            description="Model Deployment Strategy",
            goal="Deploy a trained LLM via API",
            output_format="JSON format with deployment recommendations"
        )
        self.model_dir = model_dir
        self.llm = llm_provider

    def execute(self) -> TaskResult:
        try:
            prompt = f"""
            You are an expert in deploying language models.
            The trained model is located in {self.model_dir}.

            Tasks:

            - Suggest the best architecture to serve this model (FastAPI, Flask, Triton, vLLM, etc.).
            - Provide the minimum required resources (RAM, VRAM, CPU).
            - Generate a code snippet to load the model and create a /generate endpoint.            
            """
            deployment_recommendations = self.llm.generate(prompt)
            
            model = AutoModelForCausalLM.from_pretrained(self.model_dir)
            tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            
            app = FastAPI()

            @app.post("/generate")
            async def generate_text(prompt: str):
                inputs = tokenizer(prompt, return_tensors="pt")
                outputs = model.generate(**inputs, max_length=200)
                return {"generated_text": tokenizer.decode(outputs[0], skip_special_tokens=True)}
            
            import uvicorn
            uvicorn.run(app, host="0.0.0.0", port=8000)
            
            return TaskResult(success=True, content="http://localhost:8000/generate")
        except Exception as e:
            return TaskResult(success=False, content=None, error=str(e))


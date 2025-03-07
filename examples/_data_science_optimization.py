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

# Task for selecting the appropriate model based on the task and data
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


# Task for tuning the hyperparameters of a model
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


# Task for evaluating the performance of a model
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


# Task for applying regularization techniques to avoid overfitting
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


async def main():
    print("\nInitializing ML/DL Optimization System...")
    test_keys = [os.getenv('API_KEY_1'), os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)

    # Instantiate tasks
    model_selection_task = ModelSelectionTask(
        description="Select the appropriate ML or DL model for a given task and dataset",
        goal="Recommend the best model type based on task complexity and data type",
        output_format="Model type recommendation",
        llm_provider=provider
    )

    hyperparameter_tuning_task = HyperparameterTuningTask(
        description="Tune hyperparameters of the selected model to optimize performance",
        goal="Suggest optimal hyperparameters and tuning methods",
        output_format="Hyperparameter tuning suggestions",
        llm_provider=provider
    )

    model_evaluation_task = ModelEvaluationTask(
        description="Evaluate the performance of the trained model",
        goal="Assess model performance based on relevant metrics",
        output_format="Performance evaluation report",
        llm_provider=provider
    )

    regularization_task = RegularizationTask(
        description="Apply regularization techniques to avoid overfitting",
        goal="Prevent overfitting and ensure generalization of the model",
        output_format="Regularization techniques suggestion",
        llm_provider=provider
    )

    # Create an agent for ML/DL optimization
    optimization_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[model_selection_task, hyperparameter_tuning_task, model_evaluation_task, regularization_task],
        llm_provider=provider
    )
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"optimization_agent": optimization_agent},
        flows={
            "default_flow": Flow(agents=["optimization_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Dataset: Image data for object classification task. Suggest the most suitable ML or DL model.",
        "Model: CNN. Dataset: Small image dataset. Tune hyperparameters to optimize model performance.",
        "Model: SVM. Dataset: Financial data. Evaluate model performance based on precision and recall.",
        "Model: Neural Network. Dataset: Large dataset with complex features. Suggest regularization techniques to prevent overfitting."
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("optimization_agent")
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

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

# Define custom task for analyzing student performance and recommending improvements
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


# Define a task to predict future difficulties and adjust the teaching plan
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


# Define a task to suggest materials and topics based on the student's needs
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


# Define a task to suggest activities for language learning based on identified weaknesses
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


async def main():
    print("\nInitializing Educational Improvement System...")
    test_keys = [os.getenv('API_KEY_1'), os.getenv('API_KEY_2')]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)

    # Instantiate tasks
    performance_task = StudentPerformanceAnalysisTask(
        description="Analyze student performance and identify learning gaps",
        goal="Provide recommendations to cover gaps in the student’s knowledge",
        output_format="Suggested topics for improvement",
        llm_provider=provider
    )

    prediction_task = FutureDifficultiesPredictionTask(
        description="Predict future learning difficulties and adjust teaching plan",
        goal="Suggest changes to the teaching plan based on predicted challenges",
        output_format="Suggested changes to the teaching plan",
        llm_provider=provider
    )

    material_task = MaterialRecommendationTask(
        description="Recommend materials and topics for further study",
        goal="Provide learning materials and topics to strengthen understanding",
        output_format="Recommended materials and topics",
        llm_provider=provider
    )

    language_task = LanguageLearningActivityTask(
        description="Suggest activities to help improve language learning",
        goal="Provide activities to address language learning difficulties",
        output_format="Recommended activities",
        llm_provider=provider
    )

    # Create an agent for educational analysis
    educational_agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[performance_task, prediction_task, material_task, language_task],
        llm_provider=provider
    )
    
    # Setup orchestrator
    orchestrator = Orchestrator(
        agents={"education_agent": educational_agent},
        flows={
            "default_flow": Flow(agents=["education_agent"], context_pass=[True])
        },
        mode=ExecutionMode.SEQUENTIAL
    )

    user_inputs = [
        "Student performance data: Grades in Math: B+, Science: A, English: C+. Suggest topics that need more focus.",
        "Student performance data: Struggled with physics and calculus in past semesters. Predict difficulties in upcoming chemistry and suggest teaching adjustments.",
        "Student has basic knowledge in Algebra. Suggest materials for improving calculus understanding.",
        "Student is learning Spanish. Based on assessments, suggest activities to improve vocabulary and grammar."
    ]
    
    # Process each input
    for user_input in user_inputs:
        print(f"\nUser Input: {user_input}")
        print("=" * 80)
        
        try:
            result = await orchestrator.process_request(user_input, flow_name='default_flow')
            
            if result["success"]:
                if isinstance(result["content"], dict):
                    content = result["content"].get("education_agent")
                    if content and content.success:
                        print("\nSuggestions for Improvement:")
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

import asyncio
import os
import sys
import colorama
from colorama import Fore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, TaskResult,
    ExecutionMode, AgentRole
)
from src.llm import GeminiProvider, GeminiConfig

# Define custom task for generating training code
class TrainingCodeTask(Task):
    def __init__(self, description: str, goal: str, output_format: str, llm_provider):
        super().__init__(description, goal, output_format)
        self.llm = llm_provider

    def execute(self, input_data: str) -> TaskResult:
        prompt = f"""
        Task: {self.description}
        Goal: {self.goal}
        
        Input Data: {input_data}
        
        Based on the provided information, you need to understand the complete physical system used. 
        Don't worry about the difficulty of development and processing; I need the system to be as realistic as possible to describe the problem requested by the user.
        
        From the equations provided, construct the best architecture, the best loss function, 
        and create a function to generate artificial data using functions like Sobol to add noise if necessary 
        and to obtain the most realistic data possible.
        
        Please provide a Python code snippet to train the model based on the provided information, including the governing equations.
        {self.output_format}
        """
        
        try:
            response = self.llm.generate(prompt)
            return TaskResult(
                success=True,
                content=response,
                metadata={"task_type": "training_code"}
            )
        except Exception as e:
            return TaskResult(
                success=False,
                content=None,
                error=str(e)
            )

async def main():
    colorama.init(autoreset=True)
    print(Fore.GREEN + "\nWelcome to the PINNeAPLE application!\n\n================================\n\n")

    # Ask if the user wants to solve a problem forward or inverse
    problem_type = input(Fore.YELLOW + "Do you want to solve a problem in a forward or inverse manner? (Type 'forward' or 'inverse'): \n").strip().lower()
    
    if problem_type == 'inverse' or problem_type == 'inv':
        print(Fore.RED + "This feature is still under development.\n")
        return

    elif problem_type == 'forward' or problem_type == 'fwd':
        print(Fore.GREEN + "\nProceeding with forward problem solving settings.\n")
        
        # New question about searching for existing solutions
        search_solution = input(Fore.YELLOW + "\nDo you want to search for existing solutions to this problem? (Type 'yes' or 'no'): \n").strip().lower()
        
        # New question about the framework
        framework_choice = input(Fore.YELLOW + "\nWhich framework do you want the solution in? (Type 'Pytorch', 'JAX', or 'Tensorflow'): \n").strip().lower()
        
        # New question about model specifications
        model_specifications = input(Fore.YELLOW + "\nDo you want to add any development or products specifications to the model? (Type 'yes' or 'no'): \n").strip().lower()
        
        data_physics_choice = input(Fore.YELLOW + "\nWhich option best describes your problem? (Type 'Forward Physics-Driven' or 'Forward Data-Physics-Driven'): \n").strip().lower()

        if data_physics_choice == 'forward physics-driven' or data_physics_choice == 'fpd':
            model_choice = input(Fore.YELLOW + "\nWhich model do you want to use to solve your problem? (Type 'Deep Neural Networks' or 'Extreme Learning Machine'): \n").strip().lower()

            if model_choice == 'extreme learning machine' or model_choice == 'elm':
                print(Fore.RED + "This feature is still under development.\n")
                return
            elif model_choice == 'deep neural networks' or model_choice == 'dnn':
                theory_choice = input(Fore.YELLOW + "\nDo you want to solve your problem using Theory of Functional Connections? (Type 'yes' or 'no'): \n").strip().lower()
                
                if theory_choice == 'yes' or theory_choice == 'y':
                    print(Fore.RED + "This feature is still under development.\n")
                    return
                else:
                    print(Fore.GREEN + "\nStarting problem settings for Non TFC DNN PINN model.\n")
                context = input(Fore.YELLOW + "\nPlease describe the context of the problem to be solved: \n").strip()
                print(Fore.GREEN + "\nFollowing the flow to gather geometry information.\n")
                geometry = input(Fore.YELLOW + "\nPlease describe the geometry of the problem: \n").strip()
                boundary_conditions = input(Fore.YELLOW + "\nPlease describe the boundary conditions: \n").strip()
                equations = input(Fore.YELLOW + "\nPlease provide the governing equations (if known): \n").strip()
                extra_info = input(Fore.YELLOW + "\nThere are any additional information you want to add? \n").strip()

                # Compile all information
                input_data = f"""
                Problem Context: {context}
                Geometry: {geometry}
                Boundary Conditions: {boundary_conditions}
                Governing Equations: {equations}
                Framework: {framework_choice}
                Search for Solutions: {search_solution}
                Model Specifications: {model_specifications}
                Additional Information: {extra_info}
                """
                
                print(Fore.GREEN + "\n\nThank you. Now, go drink a glass of water while we generate your training code...\n")
                await generate_training_code(input_data)

        elif data_physics_choice == 'forward data-physics-driven':
            data_file_path = input(Fore.YELLOW + "Please provide the path to your CSV or Excel file with data: ")
            input_columns = input(Fore.YELLOW + "Please specify the input columns: ")
            output_columns = input(Fore.YELLOW + "Please specify the output columns: ")

            # Compile all information
            input_data = f"""
            Data File Path: {data_file_path}
            Input Columns: {input_columns}
            Output Columns: {output_columns}
            Framework: {framework_choice}
            Search for Solutions: {search_solution}
            Model Specifications: {model_specifications}
            """
            
            print(Fore.GREEN + "\nGenerating training code...\n")
            await generate_training_code(input_data)

async def generate_training_code(input_data):
    # Initialize the LLM provider and orchestrator
    test_keys = [
        "AIzaSyCoC6voLEtOEOg5caWaqEIXBh8CiYWoUaY",
        "AIzaSyDA3r3LpI8cIGm4AVoaDQ65mDMD10GNTVM"
    ]
    
    gemini_config = GeminiConfig(
        api_keys=test_keys,
        temperature=0.7,
        verbose=False,
        max_tokens=2000
    )
    
    provider = GeminiProvider(gemini_config)
    
    # Create agent and task
    training_code_task = TrainingCodeTask(
        description="Generate Python code to train a Physics-Informed Neural Network (PINN).",
        goal="Provide a clear and functional code snippet for training the PINN.",
        output_format="Python code for training the PINN",
        llm_provider=provider
    )
    
    agent = Agent(
        role=AgentRole.SPECIALIST,
        tasks=[training_code_task],
        llm_provider=provider
    )
    
    orchestrator = Orchestrator(
        agents={"gemini": agent},
        mode=ExecutionMode.SEQUENTIAL
    )
    
    # Process the input data to generate training code
    result = await orchestrator.process_request(input_data)
    
    if result["success"]:
        if isinstance(result["content"], dict):
            content = result["content"].get("gemini")
            if content and content.success:
                training_code = content.content.strip().replace("```python", "").replace("```", "")
                print(Fore.GREEN + "\nTraining code generated successfully.\n")
                
                # Save the training code to a .py file
                with open("examples/results/training_code.py", "w") as f:
                    f.write(training_code)
                print(Fore.YELLOW + "Training code saved as 'training_code.py'.\n")

                # Gerar o código de inferência usando o código de treinamento
                inference_code = generate_inference_code(provider)
                print(Fore.GREEN + "\nInference code generated successfully.\n")
                
                # Salvar o código de inferência em um arquivo
                with open("examples/results/inference_code.py", "w") as f:
                    f.write(inference_code.replace("```python", "").replace("```", ""))
                print(Fore.YELLOW + "Inference code saved as 'inference_code.py'.\n")

                # Gerar o código de visualização usando o código de treinamento e inferência
                visualization_code = generate_visualization_code(provider)
                print(Fore.GREEN + "\nVisualization code generated successfully.\n")

                # Salvar o código de visualização em um arquivo
                with open("examples/results/visualization_code.py", "w") as f:
                    f.write(visualization_code.replace("```python", "").replace("```", ""))
                print(Fore.YELLOW + "Visualization code saved as 'visualization_code.py'.\n")

                # Gerar o README usando o LLM com todas as informações
                readme_content = generate_readme(input_data, provider)
                print(Fore.GREEN + "\nREADME generated successfully.\n")
                
                # Salvar o README gerado
                with open("examples/results/README.md", "w") as f:
                    f.write(readme_content)
                print(Fore.YELLOW + "README file created as 'README.md'.\n")

            else:
                print(Fore.RED + "Empty response received.\n")
        else:
            print(Fore.RED + "Error in response content.\n")
    else:
        print(Fore.RED + f"Error: {result['error']}\n")

def generate_inference_code(llm_provider) -> str:
    with open("examples/results/training_code.py", 'r') as f:
        training_code = f.read()
    
    prompt = f"""
    Based on the following training code, generate the inference code to run the trained model:

    Training Code:
    {training_code}

    The inference code should include:
    - Loading the trained model
    - Preparing input data for inference
    - Running inference
    - Printing or returning the output
    """

    try:
        response = llm_provider.generate(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating inference code: {str(e)}")
        return "Error generating inference code."

def generate_visualization_code(training_code: str, inference_code: str, llm_provider) -> str:
    prompt = f"""
    Based on the following training code and inference code, generate the visualization code to visualize the results of the trained model:

    Training Code:
    {training_code}

    Inference Code:
    {inference_code}

    The visualization code should include:
    - Loading the results from the inference
    - Creating visualizations (e.g., plots, graphs)
    - Displaying or saving the visualizations
    """

    try:
        response = llm_provider.generate(prompt)
        return response.text
    except Exception as e:
        print(f"Error generating visualization code: {str(e)}")
        return "Error generating visualization code."

def generate_readme(input_data: str, training_code: str, inference_code: str, visualization_code: str, llm_provider) -> str:
    prompt = f"""
    Generate a README file based on the following information:

    Problem Description:
    {input_data}

    Training Code:
    {training_code}

    Inference Code:
    {inference_code}

    Visualization Code:
    {visualization_code}

    The README should include:
    - A clear problem description
    - Instructions on how to run the training code
    - Instructions on how to run the inference code
    - Instructions on how to run the visualization code
    """

    try:
        response = llm_provider.generate(prompt)
        return response
    except Exception as e:
        print(f"Error generating README: {str(e)}")
        return "Error generating README."

if __name__ == "__main__":
    asyncio.run(main()) 
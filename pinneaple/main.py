import asyncio
import os
import sys
import colorama
from colorama import Fore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, TaskResult,
    ExecutionMode, AgentRole, Flow
)
from src.llm import GeminiProvider, GeminiConfig
from preset_agents.preset_components import create_agents, create_flows  # Updated import

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
        
        Based on the provided information, construct a robust and scalable Python code snippet to train a Physics-Informed Neural Network (PINN). 
        Ensure the code includes:
        - Best practices for model training and scientific machine learning
        - Integration with cloud services (Azure, GCP, AWS) for scalability
        - Error handling and logging mechanisms
        - Comments explaining each section of the code
        
        Please provide a Python code snippet to train the model based on the provided information, including the governing equations well defined.
        Also, create and fill all the functions and code, don't let anything open for the user.
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

    # Initialize the LLM provider
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

    # Create agents and flows
    agents = create_agents(provider)
    flows = create_flows()

    # Example usage of orchestrator
    orchestrator = Orchestrator(
        agents=agents,
        flows=flows,
        mode=ExecutionMode.SEQUENTIAL
    )

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
        flows={
            "default_flow": Flow(agents=["gemini"], context_pass=[True])  # Define a flow if needed
        },
        mode=ExecutionMode.SEQUENTIAL
    )
    
    # Process the input data to generate training code
    result = await orchestrator.process_request(input_data, flow_name="default_flow")  # Pass flow name if defined
    
    if result["success"]:
        if isinstance(result["content"], dict):
            content = result["content"].get("gemini")
            if content and content.success:
                training_code = content.content.strip().replace("```python", "").replace("```", "")
                print(Fore.GREEN + "\nTraining code generated successfully.\n")
                
                # Save the training code to a .py file
                with open("examples/results/training_code.py", "w", encoding="utf-8") as f:
                    f.write(training_code)
                print(Fore.YELLOW + "Training code saved as 'training_code.py'.\n")

                # Generate the inference code using the training code
                inference_code = generate_inference_code(provider)
                print(Fore.GREEN + "\nInference code generated successfully.\n")
                
                # Save the inference code to a file
                with open("examples/results/inference_code.py", "w", encoding="utf-8") as f:
                    f.write(inference_code.replace("```python", "").replace("```", ""))
                print(Fore.YELLOW + "Inference code saved as 'inference_code.py'.\n")

                # Generate the visualization code using the training and inference code
                visualization_code = generate_visualization_code(provider)
                print(Fore.GREEN + "\nVisualization code generated successfully.\n")

                # Save the visualization code to a file
                with open("examples/results/visualization_code.py", "w", encoding="utf-8") as f:
                    f.write(visualization_code.replace("```python", "").replace("```", ""))
                print(Fore.YELLOW + "Visualization code saved as 'visualization_code.py'.\n")

                # Generate the requirements for the project
                requirements_file = generate_requirements(provider)
                print(Fore.GREEN + "\nRequirements generated successfully.\n")

                # Save the requirements to a file
                with open("examples/results/requirements.txt", "w", encoding="utf-8") as f:
                    f.write(requirements_file)
                print(Fore.YELLOW + "Requirements saved as 'requirements.txt'.\n")

                # Generate the README using the LLM with all the information
                readme_content = generate_readme(input_data, provider)
                print(Fore.GREEN + "\nREADME generated successfully.\n")
                
                # Save the README generated
                with open("examples/results/README.md", "w", encoding="utf-8") as f:
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
    - Integration with cloud services (Azure, GCP, AWS) for deployment
    - Error handling and logging mechanisms
    - Generate data for compare, using the equations supplied before and the boundary conditions.

    The code should be concize, scalable and clean.
    Also, fill all the functions and parts of code, following scientific machine learning and development patterns.
    Don't let anything open.

    """

    try:
        response = llm_provider.generate(prompt)
        return response.replace('Inference Code:\n\n','')
    except Exception as e:
        print(f"Error generating inference code: {str(e)}")
        return "Error generating inference code."

def generate_visualization_code(llm_provider) -> str:
    with open("examples/results/training_code.py", 'r') as f:
        training_code = f.read()
    
    with open("examples/results/inference_code.py", 'r') as f:
        inference_code = f.read()
    
    prompt = f"""
    Based on the following training code and inference code, generate the best visualization code to visualize the results of the trained physics informed neural network model:

    Training Code:
    {training_code}

    Inference Code:
    {inference_code}

    The visualization code should include:
    - Loading the results from the inference
    - Creating visualizations (e.g., plots, graphs)
    - Generate data for compare, using the equations supplied before and the boundary conditions.
    - Instructions for displaying or saving the visualizations
    - Instructions for Best practices for visualization in a production environment
    
    """

    try:
        response = llm_provider.generate(prompt)
        return response.replace('```python','').replace('```','')
    except Exception as e:
        print(f"Error generating visualization code: {str(e)}")
        return "Error generating visualization code."

def generate_requirements(llm_provider) -> str:
    with open("examples/results/training_code.py", 'r') as f:
        training_code = f.read()
    
    with open("examples/results/inference_code.py", 'r') as f:
        inference_code = f.read()
    
    with open("examples/results/visualization_code.py", 'r') as f:
        visualization_code = f.read()

    prompt = f"""
    Generate a requirements.txt content based on the following information:

    Training Code:
    {training_code}

    Inference Code:
    {inference_code}

    Visualization Code:
    {visualization_code}

    The requirements should include:
    - All necessary libraries for training, inference, and visualization
    - Versions of the libraries for compatibility
    - Any additional dependencies for cloud integration (Azure, GCP, AWS)
    """

    try:
        response = llm_provider.generate(prompt)
        return response.replace('```','').replace('\nrequirements.txt\n','')
    except Exception as e:
        print(f"Error generating requirements: {str(e)}")
        return "Error generating requirements."

def generate_readme(input_data: str, llm_provider) -> str:
    with open("examples/results/training_code.py", 'r') as f:
        training_code = f.read()
    
    with open("examples/results/inference_code.py", 'r') as f:
        inference_code = f.read()
    
    with open("examples/results/visualization_code.py", 'r') as f:
        visualization_code = f.read()

    with open("examples/results/requirements.txt", 'r') as f:
        requirements_file = f.read()

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

    Requirements:
    {requirements_file}

    The README should include:
    - A clear problem description
    - Theoretycal physical and/or math explanation about the problem
    - Instructions on how to run the training code
    - Instructions on how to run the inference code
    - Instructions on how to run the visualization code
    - Instructions on how to setup and run the whole project
    - Real life applications to it
    - Best practices for deploying the model in a production environment
    - Integration with cloud services (Azure, GCP, AWS)
    
    """

    try:
        response = llm_provider.generate(prompt)
        return response
    except Exception as e:
        print(f"Error generating README: {str(e)}")
        return "Error generating README."

if __name__ == "__main__":
    asyncio.run(main()) 

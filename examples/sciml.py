import os
import sys
import asyncio
from dotenv import load_dotenv

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import (
    Agent, Task, Orchestrator, Flow,
    ExecutionMode, AgentRole, TaskResult
)
from src.llm import GeminiProvider, GeminiConfig
load_dotenv('.env')

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

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)

molecular_property_prediction_task = GeneralTask(
    description="Use machine learning techniques to predict electronic properties of molecules based on quantum chemistry calculations.",
    goal="Implement machine learning models to predict molecular electronic properties from quantum chemical data.",
    output_format="Predictions of electronic properties with corresponding accuracy metrics.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


numerical_method_comparison_task = GeneralTask(
    description="Compare the efficiency of PINNs, Fourier Neural Operators (FNO), and classical methods (FEM, FDM) for solving differential equations.",
    goal="Evaluate and compare the efficiency and accuracy of PINNs, FNO, and classical methods for solving PDEs.",
    output_format="Comparison of performance metrics for each method.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


dimensionality_reduction_task = GeneralTask(
    description="Use variational autoencoders to reduce the dimensionality of problems related to partial differential equations (PDEs) in engineering.",
    goal="Apply variational autoencoders to reduce the dimensionality of PDE problems while preserving essential information.",
    output_format="Reduced-dimensional representations of PDE data.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


monte_carlo_acceleration_task = GeneralTask(
    description="Use deep learning models to accelerate Monte Carlo integration calculations in statistical mechanics problems.",
    goal="Apply deep learning techniques to speed up Monte Carlo integration in statistical simulations.",
    output_format="Accelerated Monte Carlo integration results with reduced computational time.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


reinforcement_learning_optimization_task = GeneralTask(
    description="Use reinforcement learning to optimize adaptive schemes in finite difference methods.",
    goal="Optimize adaptive finite difference schemes using reinforcement learning algorithms.",
    output_format="Optimized parameters for finite difference schemes and their corresponding performance metrics.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


synthetic_sample_generation_task = GeneralTask(
    description="Use GANs or Diffusion Models to generate synthetic samples of rare physical phenomena in simulations.",
    goal="Generate synthetic data representing rare physical phenomena using GANs or Diffusion Models.",
    output_format="Synthetic datasets that represent rare events for further analysis.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


numerical_integration_comparison_task = GeneralTask(
    description="Compare the precision and efficiency of numerical integration methods such as Simpson’s rule, Gauss-Legendre, and Monte Carlo for different functions.",
    goal="Evaluate and compare the accuracy and computational efficiency of various numerical integration methods.",
    output_format="Performance comparison of integration methods for different functions.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


linear_system_optimization_task = GeneralTask(
    description="Optimize the resolution of sparse linear systems using iterative methods like GMRES, BiCGSTAB, and multigrid preconditioners.",
    goal="Implement and compare iterative methods for solving large sparse linear systems efficiently.",
    output_format="Optimized algorithms for sparse linear system resolution with performance metrics.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


spectral_decomposition_task = GeneralTask(
    description="Apply spectral decomposition methods such as Lanczos and Arnoldi to find eigenvalues of large sparse matrices.",
    goal="Use Lanczos and Arnoldi methods to extract eigenvalues from large sparse matrices efficiently.",
    output_format="Calculated eigenvalues and insights into the matrix structure.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


seismic_wave_simulation_task = GeneralTask(
    description="Use finite difference and finite element methods to simulate the propagation of seismic waves.",
    goal="Simulate seismic wave propagation using FDM and FEM for accurate modeling of wave behavior.",
    output_format="Seismic wave propagation simulations with corresponding data.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


thermal_conduction_task = GeneralTask(
    description="Use finite volume methods (FVM) and finite difference methods (FDM) to solve thermal conduction equations in heterogeneous media.",
    goal="Model heat transfer in heterogeneous media using FVM and FDM for accurate simulations.",
    output_format="Thermal conduction simulations with detailed temperature distributions.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


mhd_simulation_task = GeneralTask(
    description="Implement numerical simulations for magnetohydrodynamics (MHD) equations in nuclear fusion reactors.",
    goal="Simulate the behavior of plasma and magnetic fields in fusion reactors using MHD equations.",
    output_format="MHD simulation results relevant to nuclear fusion processes.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


high_dimensional_data_reduction_task = GeneralTask(
    description="Use SVD, PCA, and autoencoders to reduce high-dimensional data without losing critical information.",
    goal="Apply dimensionality reduction techniques to large datasets while maintaining key features.",
    output_format="Reduced dataset with preserved essential features.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


experimental_data_interpolation_task = GeneralTask(
    description="Compare cubic splines, Lagrange interpolation, and regression neural networks for approximating experimental data.",
    goal="Evaluate and compare interpolation methods for experimental data approximation.",
    output_format="Comparison of interpolation techniques and their performance.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


scaling_simulations_task = GeneralTask(
    description="Identify techniques for scaling numerical simulations to exascale supercomputers.",
    goal="Develop strategies for scaling numerical simulations efficiently on exascale systems.",
    output_format="Optimized scaling methods for exascale simulation environments.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


multi_criteria_optimization_task = GeneralTask(
    description="Use NSGA-II and SPEA2 algorithms to optimize simulations involving multiple criteria in engineering.",
    goal="Implement multi-objective optimization algorithms for engineering simulations with competing criteria.",
    output_format="Optimized solutions based on multi-objective criteria.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


biological_nn_modeling_task = GeneralTask(
    description="Model biological neural networks using stochastic differential equations and Hodgkin-Huxley models.",
    goal="Simulate biological neural networks using mathematical models of neuron activity.",
    output_format="Simulations of biological neural networks and their dynamics.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


genetic_circuit_simulation_task = GeneralTask(
    description="Simulate synthetic genetic circuits using mathematical modeling and cellular automata-based computation.",
    goal="Model and simulate the behavior of synthetic genetic circuits.",
    output_format="Genetic circuit simulation results with associated parameters.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


bridge_deformation_simulation_task = GeneralTask(
    description="Use the finite element method (FEM) to simulate bridge deformations under various dynamic loads.",
    goal="Simulate and analyze bridge deformations under dynamic loading conditions using FEM.",
    output_format="Simulation of bridge deformation and stress distribution.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


turbulent_flow_simulation_task = GeneralTask(
    description="Optimize turbulent flow simulations in automotive aerodynamics using turbulence models like k-ε, LES, and DNS.",
    goal="Apply turbulence models to simulate and optimize aerodynamic flows in automotive design.",
    output_format="Turbulent flow simulations with performance metrics and optimization results.",
    llm_provider=provider
)

test_keys = [os.getenv('API_KEY')]

# Set up Gemini configuration
gemini_config = GeminiConfig(
    api_keys=test_keys,
    temperature=0.7,
    verbose=False,
    max_tokens=10000
)
provider = GeminiProvider(gemini_config)


mechanical_failure_prediction_task = GeneralTask(
    description="Predict mechanical failures in metal alloys using discrete element method (DEM) simulations and machine learning.",
    goal="Model and predict mechanical failure in alloys using DEM and ML algorithms.",
    output_format="Prediction of failure points in materials and their characteristics.",
    llm_provider=provider
)

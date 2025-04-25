from presets.tasks_objects import *
from presets.providers import gemini_provider, gemini_provider2

provider = gemini_provider()
provider2 = gemini_provider2()

molecular_property_prediction_task = GeneralTask(
    description="Use machine learning techniques to predict electronic properties of molecules based on quantum chemistry calculations.",
    goal="Implement machine learning models to predict molecular electronic properties from quantum chemical data.",
    output_format="Predictions of electronic properties with corresponding accuracy metrics.",
    llm_provider=provider
)

numerical_method_comparison_task = GeneralTask(
    description="Compare the efficiency of PINNs, Fourier Neural Operators (FNO), and classical methods (FEM, FDM) for solving differential equations.",
    goal="Evaluate and compare the efficiency and accuracy of PINNs, FNO, and classical methods for solving PDEs.",
    output_format="Comparison of performance metrics for each method.",
    llm_provider=provider
)

debugging_task = GeneralTask(
    description="Analyze system logs and code execution traces to identify and fix bugs in the given code.",
    goal="Automatically detect errors in the provided code based on system logs and suggest corrections.",
    output_format="Corrected version of the code with explanations of the fixes.",
    llm_provider=provider
)

dimensionality_reduction_task = GeneralTask(
    description="Use variational autoencoders to reduce the dimensionality of problems related to partial differential equations (PDEs) in engineering.",
    goal="Apply variational autoencoders to reduce the dimensionality of PDE problems while preserving essential information.",
    output_format="Reduced-dimensional representations of PDE data.",
    llm_provider=provider
)

monte_carlo_acceleration_task = GeneralTask(
    description="Use deep learning models to accelerate Monte Carlo integration calculations in statistical mechanics problems.",
    goal="Apply deep learning techniques to speed up Monte Carlo integration in statistical simulations.",
    output_format="Accelerated Monte Carlo integration results with reduced computational time.",
    llm_provider=provider
)

reinforcement_learning_optimization_task = GeneralTask(
    description="Use reinforcement learning to optimize adaptive schemes in finite difference methods.",
    goal="Optimize adaptive finite difference schemes using reinforcement learning algorithms.",
    output_format="Optimized parameters for finite difference schemes and their corresponding performance metrics.",
    llm_provider=provider
)

synthetic_sample_generation_task = GeneralTask(
    description="Use GANs or Diffusion Models to generate synthetic samples of rare physical phenomena in simulations.",
    goal="Generate synthetic data representing rare physical phenomena using GANs or Diffusion Models.",
    output_format="Synthetic datasets that represent rare events for further analysis.",
    llm_provider=provider
)

numerical_integration_comparison_task = GeneralTask(
    description="Compare the precision and efficiency of numerical integration methods such as Simpson’s rule, Gauss-Legendre, and Monte Carlo for different functions.",
    goal="Evaluate and compare the accuracy and computational efficiency of various numerical integration methods.",
    output_format="Performance comparison of integration methods for different functions.",
    llm_provider=provider
)

linear_system_optimization_task = GeneralTask(
    description="Optimize the resolution of sparse linear systems using iterative methods like GMRES, BiCGSTAB, and multigrid preconditioners.",
    goal="Implement and compare iterative methods for solving large sparse linear systems efficiently.",
    output_format="Optimized algorithms for sparse linear system resolution with performance metrics.",
    llm_provider=provider
)

spectral_decomposition_task = GeneralTask(
    description="Apply spectral decomposition methods such as Lanczos and Arnoldi to find eigenvalues of large sparse matrices.",
    goal="Use Lanczos and Arnoldi methods to extract eigenvalues from large sparse matrices efficiently.",
    output_format="Calculated eigenvalues and insights into the matrix structure.",
    llm_provider=provider
)

seismic_wave_simulation_task = GeneralTask(
    description="Use finite difference and finite element methods to simulate the propagation of seismic waves.",
    goal="Simulate seismic wave propagation using FDM and FEM for accurate modeling of wave behavior.",
    output_format="Seismic wave propagation simulations with corresponding data.",
    llm_provider=provider
)

thermal_conduction_task = GeneralTask(
    description="Use finite volume methods (FVM) and finite difference methods (FDM) to solve thermal conduction equations in heterogeneous media.",
    goal="Model heat transfer in heterogeneous media using FVM and FDM for accurate simulations.",
    output_format="Thermal conduction simulations with detailed temperature distributions.",
    llm_provider=provider
)

mhd_simulation_task = GeneralTask(
    description="Implement numerical simulations for magnetohydrodynamics (MHD) equations in nuclear fusion reactors.",
    goal="Simulate the behavior of plasma and magnetic fields in fusion reactors using MHD equations.",
    output_format="MHD simulation results relevant to nuclear fusion processes.",
    llm_provider=provider
)

high_dimensional_data_reduction_task = GeneralTask(
    description="Use SVD, PCA, and autoencoders to reduce high-dimensional data without losing critical information.",
    goal="Apply dimensionality reduction techniques to large datasets while maintaining key features.",
    output_format="Reduced dataset with preserved essential features.",
    llm_provider=provider
)

experimental_data_interpolation_task = GeneralTask(
    description="Compare cubic splines, Lagrange interpolation, and regression neural networks for approximating experimental data.",
    goal="Evaluate and compare interpolation methods for experimental data approximation.",
    output_format="Comparison of interpolation techniques and their performance.",
    llm_provider=provider
)

scaling_simulations_task = GeneralTask(
    description="Identify techniques for scaling numerical simulations to exascale supercomputers.",
    goal="Develop strategies for scaling numerical simulations efficiently on exascale systems.",
    output_format="Optimized scaling methods for exascale simulation environments.",
    llm_provider=provider
)

multi_criteria_optimization_task = GeneralTask(
    description="Use NSGA-II and SPEA2 algorithms to optimize simulations involving multiple criteria in engineering.",
    goal="Implement multi-objective optimization algorithms for engineering simulations with competing criteria.",
    output_format="Optimized solutions based on multi-objective criteria.",
    llm_provider=provider
)

biological_nn_modeling_task = GeneralTask(
    description="Model biological neural networks using stochastic differential equations and Hodgkin-Huxley models.",
    goal="Simulate biological neural networks using mathematical models of neuron activity.",
    output_format="Simulations of biological neural networks and their dynamics.",
    llm_provider=provider
)

genetic_circuit_simulation_task = GeneralTask(
    description="Simulate synthetic genetic circuits using mathematical modeling and cellular automata-based computation.",
    goal="Model and simulate the behavior of synthetic genetic circuits.",
    output_format="Genetic circuit simulation results with associated parameters.",
    llm_provider=provider
)

bridge_deformation_simulation_task = GeneralTask(
    description="Use the finite element method (FEM) to simulate bridge deformations under various dynamic loads.",
    goal="Simulate and analyze bridge deformations under dynamic loading conditions using FEM.",
    output_format="Simulation of bridge deformation and stress distribution.",
    llm_provider=provider
)

turbulent_flow_simulation_task = GeneralTask(
    description="Optimize turbulent flow simulations in automotive aerodynamics using turbulence models like k-ε, LES, and DNS.",
    goal="Apply turbulence models to simulate and optimize aerodynamic flows in automotive design.",
    output_format="Turbulent flow simulations with performance metrics and optimization results.",
    llm_provider=provider
)

mechanical_failure_prediction_task = GeneralTask(
    description="Predict mechanical failures in metal alloys using discrete element method (DEM) simulations and machine learning.",
    goal="Model and predict mechanical failure in alloys using DEM and ML algorithms.",
    output_format="Prediction of failure points in materials and their characteristics.",
    llm_provider=provider
)

security_scraping = SecurityTask(
    description="Security Scraping",
    goal="Perform security-related scraping on the specified URL to extract potential vulnerabilities",
    output_format="HTML content with security-related elements",
    llm_provider=provider
)

vulnerability_analysis = SecurityTask(
    description="Vulnerability Analysis",
    goal="Analyze collected data for potential vulnerabilities like SQL Injection, XSS, etc.",
    output_format="List of vulnerabilities detected",
    llm_provider=provider
)

security_report = SecurityTask(
    description="Security Report",
    goal="Generate a detailed security report based on the analysis results",
    output_format="Formatted security report",
    llm_provider=provider
)

performance_analysis = MatchAnalysisTask(
    description="Performance Analysis",
    goal="Analyze player performance based on real-time stats",
    output_format="JSON format with performance insights and improvement suggestions",
    llm_provider=provider
)

fatigue_detection = MatchAnalysisTask(
    description="Fatigue Detection",
    goal="Detect player fatigue and suggest adjustments",
    output_format="JSON format with player fatigue levels and recommended actions",
    llm_provider=provider
)

tactical_adjustment = MatchAnalysisTask(
    description="Tactical Adjustment",
    goal="Optimize team tactics based on match data",
    output_format="JSON format with suggested tactical changes",
    llm_provider=provider
)

document_analysis = AnalysisTask(
    description="Document Analysis",
    goal="Extract relevant content from PDF or PPT documents",
    output_format="JSON format with extracted content",
    llm_provider=provider
)

question_analysis = AnalysisTask(
    description="Question Analysis",
    goal="Interpret the user's question and prepare it for content search",
    output_format="JSON format with the question and intended answer structure",
    llm_provider=provider
)

answer_search = AnalysisTask(
    description="Answer Search",
    goal="Search for the answer within the extracted content from the document",
    output_format="JSON format with the search result",
    llm_provider=provider
)

response_generation = AnalysisTask(
    description="Response Generation",
    goal="Generate a final response based on the found answer and provide it in a clear format",
    output_format="Final answer in text format",
    llm_provider=provider
)

feature_selection_task = TemporalSeriesForecasting(
    description="Feature Selection for Model Building",
    goal="Analyze dataset and determine which features are most relevant for modeling based on the target variable",
    output_format="List of features to use in the model",
    llm_provider=provider
)

signal_analysis_task = TemporalSeriesForecasting(
    description="Signal and Autoregressive Analysis",
    goal="Create models to analyze seasonality and trend in time series data",
    output_format="Python code to create AR or other models for time series analysis",
    llm_provider=provider
)

residual_evaluation_task = TemporalSeriesForecasting(
    description="Residual Evaluation for Signal Model",
    goal="Create code to evaluate the residuals of the signal model",
    output_format="Python code to evaluate residuals of the time series model",
    llm_provider=provider
)

lstm_model_task = TemporalSeriesForecasting(
    description="Build LSTM Model for Time Series Prediction",
    goal="Build a Long Short-Term Memory model for time series prediction based on the dataset",
    output_format="Python code for LSTM model with relevant hyperparameters",
    llm_provider=provider
)

lstm_residual_evaluation_task = TemporalSeriesForecasting(
    description="Residual Evaluation for LSTM Model",
    goal="Create code to evaluate the residuals of the LSTM model",
    output_format="Python code to evaluate residuals of the LSTM model",
    llm_provider=provider
)

neural_ode_task = TemporalSeriesForecasting(
    description="Build Neural ODE Model for Time Series Prediction",
    goal="Build a Neural ODE model for time series prediction based on the dataset",
    output_format="Python code for LSTM model with relevant hyperparameters",
    llm_provider=provider
)

neural_ode_evaluation_task = TemporalSeriesForecasting(
    description="Residual Evaluation for Neural ODE Model",
    goal="Create code to evaluate the residuals of the Neural ODE model",
    output_format="Python code to evaluate residuals of the Neural ODE model",
    llm_provider=provider
)

destination_selection_task = TravelTask(
    description="Destination Selection",
    goal="Suggest a travel destination",
    output_format="City, Country, and brief description",
    llm_provider=provider
)

budget_estimation_task = TravelTask(
    description="Budget Estimation",
    goal="Estimate travel costs",
    output_format="Breakdown of expenses",
    llm_provider=provider
)

itinerary_planning_task = TravelTask(
    description="Itinerary Planning",
    goal="Create a travel schedule",
    output_format="Day-wise activity list",
    llm_provider=provider
)

requirements_analysis = UnityCodeGenerator(
    description="Analyze Requirements",
    goal="Extract requirements for VR/AR Unity project",
    output_format="List of necessary Unity packages, SDKs, and scene components",
    llm_provider=provider
)

architecture_planning = UnityCodeGenerator(
    description="Plan Architecture",
    goal="Define the Unity scene structure and C# script organization",
    output_format="Detailed step-by-step guide on how the code should be structured",
    llm_provider=provider
)

code_generation = UnityCodeGenerator(
    description="Generate Unity C# Code",
    goal="Generate complete and well-documented Unity C# scripts",
    output_format="Complete Unity C# script",
    llm_provider=provider
)

equation_solver_task = GeneralTask(
    description="Solving Differential Equations",
    goal="Find analytical or numerical solutions to PDEs",
    output_format="Mathematical expressions or Python/NumPy code",
    llm_provider=provider
)

pinn_generation_task = GeneralTask(
    description="Generating Physics-Informed Neural Networks",
    goal="Create a neural network architecture tailored to solve PDEs",
    output_format="PyTorch model architecture",
    llm_provider=provider
)

hyperparameter_optimization_task = GeneralTask(
    description="Hyperparameter Tuning for PINNs",
    goal="Find optimal training parameters for PINN models",
    output_format="Dictionary of best hyperparameters",
    llm_provider=provider
)

orchestrator_task = GeneralTask(
    description="Orchestrating Adaptive Flow",
    goal="Determine which agent should handle the next part of the user input",
    output_format="Name of the next agent to handle the request",
    llm_provider=provider
)

validator_task = GeneralTask(
    description="Validating Adaptive Flow Response",
    goal="Determine if the agent's response is sufficient, or if another agent is needed",
    output_format="Decision ('complete' or the next agent's name)",
    llm_provider=provider
)

html_analysis_task = WebScraping(
    description="Find Necessary Data",
    goal="Retrieve all information about which HTML part is the information needed",
    output_format="Concise and informative",
    llm_provider=provider
)

scraper_generation = WebScraping(
    description="Code Generator",
    goal="Based on the HTML struture and on where the data is, create a python code to scrape that data and store in a file.",
    output_format="Complete, concize and documentated python code",
    llm_provider=provider
)
    
pinn_tuning_task = PinnHyperparameterTuningTask(
    description="Suggest adjustments to hyperparameters for PINN training",
    goal="Suggest optimal hyperparameter settings for better PINN training performance",
    output_format="Suggested hyperparameter changes",
    llm_provider=provider
)

qa_task = LLMTask(
    description="Question Answering",
    goal="Provide clear and accurate responses",
    output_format="Concise and informative",
    llm_provider=provider
)

summarization_task = LLMTask(
    description="Text Summarization",
    goal="Summarize lengthy content into key points",
    output_format="Bullet points or short paragraph",
    llm_provider=provider
)

paper_summarization = SummarizationTask(
    description="Scientific Paper Summarization",
    goal="Summarize key points of a scientific paper",
    output_format="Bullet points highlighting main findings",
    llm_provider=provider
)

linkedin_post_generation = SummarizationTask(
    description="LinkedIn Post Generation",
    goal="Create an engaging LinkedIn post based on a scientific paper summary",
    output_format="A LinkedIn-friendly post with hashtags and a call to action",
    llm_provider=provider
)

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

audience_research = EmailTask(
    description="Audience Research",
    goal="Analyze target audience and suggest the best email tone and approach",
    output_format="JSON format with audience insights and tone suggestions",
    llm_provider=provider
)

email_generation = EmailTask(
    description="Email Content Generation",
    goal="Generate an email draft based on the campaign and audience",
    output_format="Structured email content including subject line, body, and CTA",
    llm_provider=provider
)

email_optimization = EmailTask(
    description="Email Optimization",
    goal="Refine email content for clarity, engagement, and conversion",
    output_format="Optimized email content with persuasive elements",
    llm_provider=provider
)

email_personalization = EmailTask(
    description="Email Personalization",
    goal="Adapt email content for different audience segments",
    output_format="Different versions of the email for specific audience segments",
    llm_provider=provider
)

data_extraction = CVTask(
    description="Data Extraction",
    goal="Extract relevant information from resumes",
    output_format="JSON format with Name, Experience, Skills, Education, and Certifications",
    llm_provider=provider
)

skill_matching = CVTask(
    description="Skill Matching",
    goal="Match extracted skills with job requirements",
    output_format="Matched and missing skills in JSON format",
    llm_provider=provider
)

cv_scoring = CVTask(
    description="CV Scoring",
    goal="Score the resume based on experience, skills, and education",
    output_format="Score out of 100",
    llm_provider=provider
)

report_generation = CVTask(
    description="Report Generation",
    goal="Generate a structured candidate evaluation report",
    output_format="PDF or Markdown format",
    llm_provider=provider
)

autism_task = AutismSupportTask(
    description="Provide support and information related to the user query.",
    goal="Offer accurate and helpful responses using his hyperfocus as a way to improve the learning path.",
    output_format="Clear, concise response",
    llm_provider=provider
)

qa_task = QuestionAnsweringTask(
    description="Answer questions using Gemini",
    goal="Provide accurate and helpful answers",
    output_format="Clear, concise response",
    llm_provider=provider
)

agent_creation_task = AgentCreationTask(
    description="Create a new agent based on user input",
    goal="Generate a new agent task description and goal",
    output_format="Task description, goal, and output format",
    llm_provider=provider
)

data_understanding_task = AnomaliesDetection(
    description="Understanding Data",
    goal="Retrieve all information about a tabular dataset",
    output_format="Concise and informative",
    llm_provider=provider
)

statistics_task = AnomaliesDetection(
    description="Statistics Analysis",
    goal="Retrieve all the Statistics and general behavior of data on dataset",
    output_format="Bullet points or short paragraph",
    llm_provider=provider
)

anomalies_detection_task = AnomaliesDetection(
    description="Outliers Pattern Analysis",
    goal="Analyze and return the interval of data which seems to have anomalies compared with the data pattern",
    output_format="Bullet points or short paragraph",
    llm_provider=provider
)

data_analysis_task = AnomaliesDetection(
    description="Code Generation for Data Analysis",
    goal="Based on the previous analysis, generate a code in python to execute all of them.",
    output_format="Documentated, concize and complete python code",
    llm_provider=provider
)

problem_analysis_task = ProblemAnalysis(
    description="Problem Analysis for FEM/FVM/FEA",
    goal="Analyze the problem and determine which approach (FEM, FVM, or FEA) is suitable for the given input.",
    output_format="Detailed analysis with methodology selection",
    llm_provider=provider
)

numerical_modeling_task = ProblemAnalysis(
    description="Solve the problem using the method recommended",
    goal="Solve the problem based on recommended method, using Python",
    output_format="Documented and full python code",
    llm_provider=provider
)

pinn_modeling_task = PINNModeling(
    description="PINN Modeling",
    goal="Model a Physics Informed Neural Network to solve the problem and compare it with FEM/FVM/FEA results.",
    output_format="Complete Python code to build and solve the PINN, with comparison of results",
    llm_provider=provider
)

motion_analysis_task = GeneralTask(
    description="Analyze stellar motion data to identify stars with high velocity moving toward the center of the Milky Way.",
    goal="Identify stars exhibiting high proper motion and radial velocity directed toward the galactic center.",
    output_format="List of stars with high velocity toward the center of the Milky Way.",
    llm_provider=provider
)

photometric_classification_task = GeneralTask(
    description="Classify stars based on their BP and RP photometric data, distinguishing types like red dwarfs, giants, and others.",
    goal="Identify different stellar types based on their temperature and composition using Gaia's photometric data.",
    output_format="Classification of stars into categories such as red dwarfs, giants, etc.",
    llm_provider=provider
)

galactic_dynamics_task = GeneralTask(
    description="Analyze stellar velocity and proper motion data to understand the movement of stars relative to the Sun and the rotation of the Milky Way.",
    goal="Study stellar velocity distributions to investigate the dynamics of the Milky Way, including star movements relative to the galactic center.",
    output_format="Understanding of the rotation of the Milky Way and stellar movement patterns.",
    llm_provider=provider
)

galactic_structure_task = GeneralTask(
    description="Study the distribution of stars in the Milky Way and identify structures such as spiral arms and star-forming regions.",
    goal="Map stellar distribution and identify spiral arm patterns and star-forming regions in the Milky Way.",
    output_format="Maps of stellar distribution and identification of galactic structures.",
    llm_provider=provider
)

stellar_variability_task = GeneralTask(
    description="Detect variable stars and explosive events such as supernovae based on their light curves.",
    goal="Identify variable stars and supernova events by analyzing changes in their brightness over time.",
    output_format="Classification of variable stars and detection of explosive events.",
    llm_provider=provider
)

chemical_composition_task = GeneralTask(
    description="Analyze the spectroscopic data of stars in the halo of the Milky Way to determine their chemical composition.",
    goal="Study the metallicity and chemical patterns of stars in the Milky Way’s halo to understand its formation and evolution.",
    output_format="Determination of stellar chemical composition and insights into galactic evolution.",
    llm_provider=provider
)

exoplanet_detection_task = GeneralTask(
    description="Identify stars with potential exoplanets based on astrometric data and radial velocity measurements.",
    goal="Use astrometric data to detect stars with exoplanets and calculate their orbits.",
    output_format="List of stars with exoplanet candidates and calculated orbital parameters.",
    llm_provider=provider
)

binary_system_analysis_task = GeneralTask(
    description="Study the dynamics of binary star systems using motion and distance data to infer their properties and impact on star formation.",
    goal="Analyze binary star system dynamics and their implications for star formation processes.",
    output_format="Classification and properties of binary star systems and insights into their role in star formation.",
    llm_provider=provider
)

space_mission_planning_task = GeneralTask(
    description="Optimize spacecraft trajectories based on planetary orbits and celestial body positions.",
    goal="Calculate optimized trajectories for space exploration missions using planetary orbital data.",
    output_format="Optimized trajectory plan and mission parameters.",
    llm_provider=provider
)

scientific_discovery_task = GeneralTask(
    description="Suggest new research approaches based on the latest scientific publications in a given field of physics.",
    goal="Identify emerging trends and suggest new hypotheses for future research based on the latest publications.",
    output_format="Suggested hypotheses and research approaches for future investigation.",
    llm_provider=provider
)

preferences_task = CarPurchaseTask(
    description="Preferences Analysis",
    goal="Analyze the customer preferences and create a list of viable car models and options",
    output_format="JSON format with the analyzed preferences",
    llm_provider=provider
)

payment_task = CarPurchaseTask(
    description="Payment Calculation",
    goal="Calculate the payment conditions based on financing options and customer budget",
    output_format="JSON format with payment details and final amount",
    llm_provider=provider
)

proposal_task = CarPurchaseTask(
    description="Proposal Generation",
    goal="Generate a personalized proposal with the final price, payment options, and accessories included",
    output_format="Detailed proposal with car model, payment terms, accessories, and total cost",
    llm_provider=provider
)

review_task = CarPurchaseTask(
    description="Proposal Review",
    goal="Review the proposal to ensure it is clear, concise, and covers all customer needs",
    output_format="Clear and final version of the proposal",
    llm_provider=provider
)

turbulence_modeling = GeneralTask(
    description="Apply machine learning techniques to improve turbulence models in fluid dynamics.",
    goal="Enhance turbulence predictions by integrating data-driven approaches with traditional physics-based models.",
    output_format="Improved turbulence models with reduced computational cost and higher accuracy.",
    llm_provider=provider
)

surrogate_modeling = GeneralTask(
    description="Develop machine learning-based surrogate models for expensive physics simulations.",
    goal="Replace computationally intensive simulations with fast, approximate models that retain accuracy.",
    output_format="Trained surrogate models and comparison with full-scale simulations.",
    llm_provider=provider
)

uncertainty_quantification = GeneralTask(
    description="Use probabilistic machine learning methods to quantify uncertainties in physics-based models.",
    goal="Improve the reliability of scientific predictions by characterizing uncertainty in simulation outputs.",
    output_format="Uncertainty estimates and confidence intervals for model predictions.",
    llm_provider=provider
)

inverse_problem_solver = GeneralTask(
    description="Apply machine learning techniques to solve inverse problems in physics and engineering.",
    goal="Recover hidden parameters or physical properties from observational data.",
    output_format="Reconstructed parameters or system properties with uncertainty quantification.",
    llm_provider=provider
)

ai_materials_discovery = GeneralTask(
    description="Use deep learning to predict new materials with desirable properties for applications like superconductors or semiconductors.",
    goal="Accelerate materials discovery by replacing traditional trial-and-error methods with AI-driven approaches.",
    output_format="List of candidate materials and their predicted properties.",
    llm_provider=provider
)

neural_operators_modeling = GeneralTask(
    description="Develop deep learning models capable of learning operators for solving differential equations efficiently.",
    goal="Create generalized neural network architectures that can solve a variety of PDEs with minimal computational cost.",
    output_format="Trained neural operators and benchmarking results against traditional solvers.",
    llm_provider=provider
)

cfd_modelling = GeneralTask(
    description="Based on an specific framework, generate a solution for a cfd problem.",
    goal="You need to create a documentate, complete and clear python code for solving the cfd problem with the required framework.",
    output_format="A clear, complete and documentated python code.",
    llm_provider=provider
)

solver_task = GeneralTask(
    description="Solve the full-order dynamical system using OpInf.",
    goal="Obtain a numerical solution to the original system before model reduction.",
    output_format="Solution trajectory of the full-order system.",
    llm_provider=provider
)

reduction_task = GeneralTask(
    description="Apply Operator Inference to construct a reduced-order model (ROM).",
    goal="Generate a polynomial ROM using OpInf to approximate the original system.",
    output_format="Reduced-order system representation and learned operators.",
    llm_provider=provider
)

optimization_task = GeneralTask(
    description="Optimize hyperparameters of the reduced-order model.",
    goal="Improve the accuracy and efficiency of the OpInf-generated ROM.",
    output_format="Optimized parameters and performance evaluation.",
    llm_provider=provider
)

product_manager_task = LLMTask(
    description="Product Definition",
    goal="Define target users, features, and core problem to solve",
    output_format="""
    Personas:
    - [Persona descriptions]

    Core Features:
    - [List of features]

    Problem Being Solved:
    - [Problem description]
    """,
    llm_provider=provider
)

dev_agent_task = LLMTask(
    description="Fullstack Application: FastAPI + React Integration",
    goal="""
    Generate a working fullstack boilerplate app with a FastAPI backend and React (Vite) frontend. 
    The app must include minimal setup and demonstrate integration by calling a backend route from the frontend.

    Requirements:
    - Backend must be a FastAPI app with at least two routes: "/" and "/api/hello"
    - Frontend must use React with Vite, and call the "/api/hello" route
    - The frontend and backend should be set up to run in dev mode together (via proxy or CORS)
    - Provide instructions to run both frontend and backend
    - Keep it simple: no database or authentication

    Deliver the following file structure and content:
    """,
    output_format="""
    **backend/main.py**
    [FastAPI app with "/" and "/api/hello" routes, CORS enabled for frontend]

    **backend/requirements.txt**
    [fastapi, uvicorn, python-multipart, and fastapi-cors if necessary]

    **frontend/vite.config.js**
    [Configure Vite to proxy /api requests to backend on localhost:8000]

    **frontend/public/index.html**
    [HTML entry point with <div id="root">]

    **frontend/src/index.js**
    [React entry point rendering <App />]

    **frontend/src/App.jsx**
    [Main component with route to Home]

    **frontend/src/pages/Home.jsx**
    [Component that fetches data from /api/hello and displays it]

    **frontend/src/services/api.js**
    [Function using fetch or axios to get data from /api/hello]

    **frontend/package.json**
    [React + Vite setup with scripts: "dev", "build", "preview"]

    **README.md**
    [Instructions to run the backend (`uvicorn main:app`) and frontend (`npm install && npm run dev`)]

    """,
    llm_provider=provider2
)

ui_designer_task = LLMTask(
    description="UI Mockup",
    goal="Describe UI structure and logic",
    output_format="""
    UI Layout Description:
    - [Component list and layout]

    User Flow:
    - [Sequence of interactions]

    Style Notes:
    - [Colors, fonts, buttons, etc.]
    """,
    llm_provider=provider
)

growth_hacker_task = LLMTask(
    description="Marketing Plan",
    goal="Plan product launch strategy and growth channels",
    output_format="""
    Launch Strategy:
    - [Steps for launch]

    Channels:
    - [Marketing & acquisition channels]

    Early Metrics to Monitor:
    - [KPIs to track growth]
    """,
    llm_provider=provider
)

get_requirements_task = LLMTask(
    description="Requirements Gathering for MVP",
    goal="""
    Generate a clear and actionable list of functional and non-functional requirements for a minimum viable product (MVP) web application.

    The requirements must be based on a user-provided idea, covering key features, user roles, and essential use cases.

    Requirements:
    - Identify and separate functional and non-functional requirements
    - Include assumptions and constraints
    - Present user roles and expected behaviors
    - Outline main use cases with brief descriptions
    - Output must be structured and actionable to guide development
    
    """,
    output_format="""
    **Functional Requirements**
    [Bullet list of key app features]

    **Non-Functional Requirements**
    [Performance, scalability, maintainability, etc.]

    **User Roles**
    [Roles like admin, user, etc. and what each can do]

    **Use Cases**
    [Brief descriptions like: "User can register and login", "Admin can manage content"]

    **Assumptions and Constraints**
    [E.g., "App will be web-only for MVP", "No login via social media"]

    """,
    llm_provider=provider2
)

backend_dev_task = LLMTask(
    description="Backend Development for MVP with FastAPI",
    goal="""
    Develop a functional backend for the MVP using FastAPI. The backend should include core business logic based on the previously defined requirements and support basic CRUD operations.

    Requirements:
    - Use FastAPI with appropriate Pydantic models and route definitions
    - Include routes for creating, reading, updating, and deleting key resources
    - Implement proper status codes and error handling
    - Add CORS middleware to allow frontend access
    - Use simple in-memory storage or Python lists/dicts (no DB needed for MVP)

    Deliver the following:

    """,
    output_format="""
    **backend/main.py**
    [Main FastAPI app with all defined endpoints]

    **backend/models.py**
    [Pydantic schemas for request/response data]

    **backend/routes.py**
    [Router logic for endpoints like /items, /users, etc.]

    **backend/utils.py**
    [Any helper logic or in-memory storage]

    **backend/requirements.txt**
    [List of Python dependencies: fastapi, uvicorn, etc.]

    **README.md**
    [How to run backend: `uvicorn main:app --reload`]

    """,
    llm_provider=provider2
)

frontend_dev_task = LLMTask(
    description="Frontend MVP Development with React (Vite) Integrated to FastAPI",
    goal="""
    Create a minimal React frontend using Vite that communicates with the FastAPI backend developed earlier. Demonstrate fetching and displaying data from the backend, and allow user interaction with it (e.g., creating or deleting an item).

    Requirements:
    - Use React with Vite
    - Create pages for listing, creating, and deleting items
    - Call backend endpoints using fetch or axios
    - Configure proxy or use CORS to enable integration
    - Ensure responsive and user-friendly layout

    Deliver the following file structure and content:
    """,
    output_format="""
    **frontend/vite.config.js**
    [Proxy /api requests to backend]

    **frontend/public/index.html**
    [HTML file with root div]

    **frontend/src/index.js**
    [React root entry file]

    **frontend/src/App.jsx**
    [Main app routing and layout]

    **frontend/src/pages/**
    [Pages like Home.jsx, CreateItem.jsx]

    **frontend/src/components/**
    [Reusable components like ItemCard.jsx]

    **frontend/src/services/api.js**
    [API service layer for calling backend]

    **frontend/package.json**
    [Dependencies and scripts]

    **README.md**
    [How to install and run: `npm install && npm run dev`]

    """,
    llm_provider=provider2
)

cicd_task = LLMTask(
    description="CI/CD Pipeline for Fullstack FastAPI + React App",
    goal="""
    Create a CI/CD pipeline to build, test, and deploy the FastAPI + React application using GitHub Actions. Ensure automatic installation, testing, and deployment (if applicable).

    Requirements:
    - Separate jobs for backend and frontend
    - Install dependencies, run tests, and build frontend
    - Optionally deploy to platforms like Render, Vercel, or Docker container
    - Include `.github/workflows/main.yml` with all steps defined
    - Keep it simple for MVP, avoid production-grade complexity

    Deliver the following:

    """,
    output_format="""
    **.github/workflows/main.yml**
    [CI/CD pipeline steps: install, test, build, deploy]

    **Dockerfile (if used)**
    [Unified Dockerfile or separate for backend/frontend]

    **README.md**
    [How to trigger and understand the pipeline, deployment links if applicable]

    """,
    llm_provider=provider2
)

structuring_task = GeneralTask(
    description="Receive a list of texts on different formats.",
    goal="Return a JSON dictionary with the data organized in a tabular struture.",
    output_format="A structured JSON dictionary with the relevant fields extracted from the texts.",
    llm_provider=provider
)

ocr_task = GeneralTask(
    description="Receive the path to an image and use OCR to extract the visible texts from the image.",
    goal="Return only the readable texts extracted from the image, with no additional formatting or explanations.",
    output_format="A string containing the texts extracted from the image.",
    llm_provider=provider
)

// src/components/AgentCreateScreen.tsx
import React, { useState } from "react";
import axios from "axios";

interface AgentCreateScreenProps {}

const AgentCreateScreen: React.FC<AgentCreateScreenProps> = () => {
  const [agentName, setAgentName] = useState<string>("");
  const [agentDescription, setAgentDescription] = useState<string>("");
  const [selectedRole, setSelectedRole] = useState<string>("main"); // Default role
  const [selectedTasks, setSelectedTasks] = useState<string[]>([]);
  const [newTaskDescription, setNewTaskDescription] = useState<string>("");
  const [newTaskGoal, setNewTaskGoal] = useState<string>("");
  const [newTaskOutputFormat, setNewTaskOutputFormat] = useState<string>("");
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Roles (Markdown)
  const rolesMarkdown = `
    - MAIN = "main"
    - SPECIALIST = "specialist"
    - VALIDATOR = "validator"
    - REFINER = "refiner"
    - SUMMARIZER = "summarizer"
    - SCIENTIST = "scientist"
    - ENGINEER = "engineer"
    - ARCHITECT = "architect"
    - CUSTOM = "custom"
  `;

  // Preset Tasks
  const presetTasks = [
    "molecular_property_prediction_task", "numerical_method_comparison_task", "dimensionality_reduction_task",
    "monte_carlo_acceleration_task", "reinforcement_learning_optimization_task", "synthetic_sample_generation_task",
    "numerical_integration_comparison_task", "linear_system_optimization_task", "spectral_decomposition_task",
    "seismic_wave_simulation_task", "thermal_conduction_task", "mhd_simulation_task",
    "high_dimensional_data_reduction_task", "experimental_data_interpolation_task", "scaling_simulations_task",
    "multi_criteria_optimization_task", "biological_nn_modeling_task", "genetic_circuit_simulation_task",
    "bridge_deformation_simulation_task", "turbulent_flow_simulation_task", "mechanical_failure_prediction_task",
    "security_scraping", "vulnerability_analysis", "security_report", "performance_analysis", "fatigue_detection",
    "tactital_adjustment", "document_analysis", "question_analysis", "answer_search", "response_generation",
    "feature_selection_task", "signal_analysis_task", "residual_evaluation_task", "lstm_model_task",
    "lstm_residual_evaluation_task", "neural_ode_task", "neural_ode_evaluation_task", "destination_selection_task",
    "budget_estimation_task", "itinerary_planning_task", "requirements_analysis", "architecture_planning",
    "code_generation", "equation_solver_task", "pinn_generation_task", "hyperparameter_optimization_task",
    "orchestrator_task", "validator_task", "html_analysis_task", "scraper_generation", "pinn_tuning_task",
    "qa_task", "summarization_task", "paper_summarization", "linkedin_post_generation", "investment_task",
    "credit_risk_task", "portfolio_task", "fraud_detection_task", "performance_task", "prediction_task",
    "material_task", "language_task", "model_selection_task", "hyperparameter_tuning_task", "model_evaluation_task",
    "regularization_task", "audience_research", "email_generation", "email_optimization", "email_personalization",
    "data_extraction", "skill_matching", "cv_scoring", "report_generation", "autism_atsk", "agent_creation_task",
    "data_understanding_task", "statistics_task", "anomalies_detection_task", "data_analysis_task",
    "problem_analysis_task", "numerical_modeling_task", "pinn_modeling_task", "motion_analysis_task",
    "photometric_classification_task", "galactic_dynamics_task", "galactic_structure_task", "stellar_variability_task",
    "chemical_composition_task", "exoplanet_detection_task", "binary_system_analysis_task", "space_mission_planning_task",
    "scientific_discovery_task", "preferences_task", "payment_task", "proposal_task", "review_task"
  ];

  const handleCreateAgent = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await axios.post("http://127.0.0.1:8000/create-agent", {
        agent_name: agentName,
        agent_description: agentDescription,
        agent_role: selectedRole,
        agent_tasks: selectedTasks,
        new_task_description: newTaskDescription,
        new_task_goal: newTaskGoal,
        new_task_output_format: newTaskOutputFormat,
      });

      if (res.data && res.data.message) {
        setResponse(res.data.message);
      } else {
        setError("Error creating agent");
      }
    } catch (err: any) {
      setError(err.message || "Error creating agent");
    }
    setLoading(false);
  };

  const handleRoleSelect = (role: string) => {
    setSelectedRole(role);
  };

  const handleTaskSelect = (task: string) => {
    if (selectedTasks.includes(task)) {
      setSelectedTasks(selectedTasks.filter((t) => t !== task));
    } else {
      setSelectedTasks([...selectedTasks, task]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 flex items-center justify-center">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-semibold mb-4 text-center">
          Create New Agent
        </h1>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Agent Name:
          </label>
          <br></br>
          <input
            type="text"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={agentName}
            onChange={(e) => setAgentName(e.target.value)}
            placeholder="Enter agent name"
          />
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Agent Description:
          </label>
          <br></br>
          <textarea
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            rows={4}
            value={agentDescription}
            onChange={(e) => setAgentDescription(e.target.value)}
            placeholder="Enter agent description"
          />
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Agent Role:
          </label>
          <br></br>
          <select
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={selectedRole}
            onChange={(e) => handleRoleSelect(e.target.value)}
          >
            {rolesMarkdown.split('\n').map((line, index) => {
              const role = line.split('=')[1]?.trim().replace(/\"/g, '');
                 if (role) {
                   return (
                     <option key={index} value={role}>
                       {role}
                     </option>
                   );
                 }
                 return null;
               })}
             </select>
           </div>

           <div className="mb-4">
             <label className="block text-sm font-medium text-gray-700">
               Preset Tasks:
             </label>
             <div className="grid grid-cols-3 gap-4">
               {presetTasks.map((task) => (
                 <div
                   key={task}
                   className={`p-4 border rounded-md cursor-pointer ${
                     selectedTasks.includes(task)
                       ? "bg-indigo-100 border-indigo-500"
                       : "border-gray-300 hover:border-indigo-300"
                   }`}
                   onClick={() => handleTaskSelect(task)}
                 >
                   <p className="text-xs">{task}</p>
                 </div>
               ))}
             </div>
           </div>

           <div className="mb-4">
             <label className="block text-sm font-medium text-gray-700">
               Create New Task:
             </label>
             <div className="grid grid-cols-1 gap-4">
               <textarea
                 className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                 rows={2}
                 value={newTaskDescription}
                 onChange={(e) => setNewTaskDescription(e.target.value)}
                 placeholder="Task Description"
               />
               <textarea
                 className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                 rows={2}
                 value={newTaskGoal}
                 onChange={(e) => setNewTaskGoal(e.target.value)}
                 placeholder="Task Goal"
               />
               <textarea
                 className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                 rows={2}
                 value={newTaskOutputFormat}
                 onChange={(e) => setNewTaskOutputFormat(e.target.value)}
                 placeholder="Task Output Format"
               />
             </div>
           </div>

           <button
             className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 rounded-md text-white font-semibold"
             onClick={handleCreateAgent}
             disabled={loading}
           >
             {loading ? "Creating..." : "Create Agent"}
           </button>

           {error && <p className="text-red-500 mt-2 text-center">{error}</p>}

           {response && (
             <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
               <h2 className="text-lg font-semibold mb-2">Response:</h2>
               <p>{response}</p>
             </div>
           )}
         </div>
       </div>
     );
   };

   export default AgentCreateScreen;
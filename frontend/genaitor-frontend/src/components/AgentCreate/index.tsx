"use client"
import { FormEvent, useState } from "react"
import { ContentAgentCreateStyled } from "./styled"
import { Button, Form, Spinner } from "react-bootstrap"
import { service } from "../../../server"
import { toast } from "react-toastify"
import LoadingContent from "../LoadingContent"
export default function AgentCreate() {
	const [agentName, setAgentName] = useState<string>("")
	const [agentDescription, setAgentDescription] = useState<string>("")
	const [selectedRole, setSelectedRole] = useState<string>("main") // Default role
	const [selectedTasks, setSelectedTasks] = useState<string[]>([])
	const [newTaskDescription, setNewTaskDescription] = useState<string>("")
	const [newTaskGoal, setNewTaskGoal] = useState<string>("")
	const [newTaskOutputFormat, setNewTaskOutputFormat] = useState<string>("")
	const [response, setResponse] = useState<string | null>(null)
	const [loading, setLoading] = useState<boolean>(false)

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
  `

	// Preset Tasks
	const presetTasks = [
		"molecular_property_prediction_task",
		"numerical_method_comparison_task",
		"dimensionality_reduction_task",
		"monte_carlo_acceleration_task",
		"reinforcement_learning_optimization_task",
		"synthetic_sample_generation_task",
		"numerical_integration_comparison_task",
		"linear_system_optimization_task",
		"spectral_decomposition_task",
		"seismic_wave_simulation_task",
		"thermal_conduction_task",
		"mhd_simulation_task",
		"high_dimensional_data_reduction_task",
		"experimental_data_interpolation_task",
		"scaling_simulations_task",
		"multi_criteria_optimization_task",
		"biological_nn_modeling_task",
		"genetic_circuit_simulation_task",
		"bridge_deformation_simulation_task",
		"turbulent_flow_simulation_task",
		"mechanical_failure_prediction_task",
		"security_scraping",
		"vulnerability_analysis",
		"security_report",
		"performance_analysis",
		"fatigue_detection",
		"tactital_adjustment",
		"document_analysis",
		"question_analysis",
		"answer_search",
		"response_generation",
		"feature_selection_task",
		"signal_analysis_task",
		"residual_evaluation_task",
		"lstm_model_task",
		"lstm_residual_evaluation_task",
		"neural_ode_task",
		"neural_ode_evaluation_task",
		"destination_selection_task",
		"budget_estimation_task",
		"itinerary_planning_task",
		"requirements_analysis",
		"architecture_planning",
		"code_generation",
		"equation_solver_task",
		"pinn_generation_task",
		"hyperparameter_optimization_task",
		"orchestrator_task",
		"validator_task",
		"html_analysis_task",
		"scraper_generation",
		"pinn_tuning_task",
		"qa_task",
		"summarization_task",
		"paper_summarization",
		"linkedin_post_generation",
		"investment_task",
		"credit_risk_task",
		"portfolio_task",
		"fraud_detection_task",
		"performance_task",
		"prediction_task",
		"material_task",
		"language_task",
		"model_selection_task",
		"hyperparameter_tuning_task",
		"model_evaluation_task",
		"regularization_task",
		"audience_research",
		"email_generation",
		"email_optimization",
		"email_personalization",
		"data_extraction",
		"skill_matching",
		"cv_scoring",
		"report_generation",
		"autism_atsk",
		"agent_creation_task",
		"data_understanding_task",
		"statistics_task",
		"anomalies_detection_task",
		"data_analysis_task",
		"problem_analysis_task",
		"numerical_modeling_task",
		"pinn_modeling_task",
		"motion_analysis_task",
		"photometric_classification_task",
		"galactic_dynamics_task",
		"galactic_structure_task",
		"stellar_variability_task",
		"chemical_composition_task",
		"exoplanet_detection_task",
		"binary_system_analysis_task",
		"space_mission_planning_task",
		"scientific_discovery_task",
		"preferences_task",
		"payment_task",
		"proposal_task",
		"review_task",
	]

	const handleTaskSelect = (task: string) => {
		if (selectedTasks.includes(task)) {
			setSelectedTasks(selectedTasks.filter((t) => t !== task))
		} else {
			setSelectedTasks([...selectedTasks, task])
		}
	}

	const handleCreateAgent = async (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault()
		setLoading(true)
		await service
			.post("/agent", {
				agent_name: agentName,
				agent_description: agentDescription,
				agent_role: selectedRole,
				agent_tasks: selectedTasks,
				new_task_description: newTaskDescription,
				new_task_goal: newTaskGoal,
				new_task_output_format: newTaskOutputFormat,
			})
			.then((res) => {
				const { message } = res.data
				if (message) {
					setResponse(message)
				} else {
					console.log("[Error creating agent]", res.data)
					toast.error("Error creating agent")
				}
			})
			.catch((err) => {
				console.log("[Error creating agent]", err)
				toast.error("Error creating agent")
			})
			.finally(() => {
				setLoading(false)
			})
	}

	return (
		<ContentAgentCreateStyled>
			<h6>Create New Agent</h6>
			<Form id="form-agent-create" onSubmit={handleCreateAgent}>
				<div className="row">
					<div className="col-md-6 col-12 mb-3">
						<Form.Group controlId="agent-name">
							<Form.Label>Agent Name</Form.Label>
							<Form.Control
								type="text"
								placeholder="Enter agent name"
								value={agentName}
								onChange={(e) => setAgentName(e.target.value)}
							/>
						</Form.Group>
					</div>
					<div className="col-md-6 col-12 mb-3">
						<Form.Group controlId="agent-role">
							<Form.Label>Agent Role</Form.Label>
							<Form.Select
								value={selectedRole}
								onChange={(e) => setSelectedRole(e.target.value)}
								aria-label="Agent role select"
							>
								{rolesMarkdown.split("\n").map((line, index) => {
									const role = line.split("=")[1]?.trim().replace(/\"/g, "")
									if (role) {
										return (
											<option key={role + index} value={role}>
												{role}
											</option>
										)
									}
									return null
								})}
							</Form.Select>
						</Form.Group>
					</div>
				</div>
				<Form.Group controlId="agent-description">
					<Form.Label>Agent Description</Form.Label>
					<Form.Control
						as="textarea"
						rows={4}
						placeholder="Enter agent description"
						value={agentDescription}
						onChange={(e) => setAgentDescription(e.target.value)}
					/>
				</Form.Group>

				<Form.Group controlId="agent-tasks">
					<Form.Label>Preset Tasks</Form.Label>
					<div className="d-flex gap-3 flex-wrap">
						{presetTasks.map((task) => {
							return (
								<Button
									key={task}
									variant={
										selectedTasks.includes(task) ? "primary" : "outline-primary"
									}
									onClick={() => handleTaskSelect(task)}
								>
									{task}
								</Button>
							)
						})}
					</div>
				</Form.Group>
				<Form.Group controlId="new-task">
					<Form.Label>Create New Task</Form.Label>
					<div className="row">
						<div className="col-md-4 col-12 mb-3">
							<Form.Control
								as="textarea"
								rows={2}
								value={newTaskDescription}
								onChange={(e) => setNewTaskDescription(e.target.value)}
								placeholder="Task Description"
							/>
						</div>
						<div className="col-md-4 col-12 mb-3">
							<Form.Control
								as="textarea"
								rows={2}
								value={newTaskGoal}
								onChange={(e) => setNewTaskGoal(e.target.value)}
								placeholder="Task Goal"
							/>
						</div>
						<div className="col-md-4 col-12 mb-3">
							<Form.Control
								as="textarea"
								rows={2}
								value={newTaskOutputFormat}
								onChange={(e) => setNewTaskOutputFormat(e.target.value)}
								placeholder="Task Output Format"
							/>
						</div>
					</div>
				</Form.Group>
				<div className="action-button">
					<Button type="submit" disabled={loading}>
						{loading ? <Spinner size="sm" color="white" /> : "Create Agent"}
					</Button>
				</div>
			</Form>
			{loading && <LoadingContent />}
			{response && (
				<div className="response-content">
					<h6>Response:</h6>
					<p>{response}</p>
				</div>
			)}
		</ContentAgentCreateStyled>
	)
}

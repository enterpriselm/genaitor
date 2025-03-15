"use client"
import { Button, Dropdown, Form, InputGroup, Spinner } from "react-bootstrap"
import { FlowCreateContainer } from "./styled"
import { FormEvent, useState } from "react"
import LoadingContent from "../LoadingContent"
import { service } from "../../../server"
import { toast } from "react-toastify"
import { ChevronDownIcon, XIcon } from "lucide-react"
export default function FlowCreate() {
	const [flowName, setFlowName] = useState<string>("")
	const [searchValue, setSearchValue] = useState<string>("")
	const [flowDescription, setFlowDescription] = useState<string>("")
	const [selectedAgents, setSelectedAgents] = useState<string[]>([])
	const [agentSequences, setAgentSequences] = useState<{
		[key: string]: number
	}>({})
	const [flowType, setFlowType] = useState<string>("sequential")
	const [response, setResponse] = useState<string | null>(null)
	const [loading, setLoading] = useState<boolean>(false)

	// Lista de agentes (substitua com a sua lista real de agentes)
	const availableAgents = [
		"qa_agent",
		"autism_agent",
		"agent_creation",
		"data_understanding_agent",
		"statistics_agent",
		"anomalies_detection_agent",
		"data_analysis_agent",
		"problem_analysis_agent",
		"numerical_analysis_agent",
		"pinn_modeling_agent",
		"preferences_agent",
		"payment_agent",
		"proposal_agent",
		"review_agent",
		"extraction_agent",
		"matching_agent",
		"scoring_agent",
		"report_agent",
		"optimization_agent",
		"educational_agent",
		"research_agent",
		"content_agent",
		"personalization_agent",
		"financial_agent",
		"summatization_agent",
		"linkedin_agent",
		"pinn_tuning_agent",
		"html_analysis_agent",
		"scraper_generation_agent",
		"equation_solver_agent",
		"pinn_generation_agent",
		"hyperparameter_optimization_agent",
		"orchestrator_agent",
		"validator_agent",
		"requirements_agent",
		"architecture_agent",
		"code_generation_agent",
		"destination_selection_agent",
		"budget_estimation_agent",
		"itinerary_planning_agent",
		"feature_selection_agent",
		"signal_analysis_agent",
		"residual_evaluation_agent",
		"lstm_model_agent",
		"lstm_residual_evaluation_agent",
		"document_agent",
		"question_agent",
		"search_agent",
		"response_agent",
		"performance_agent",
		"fatigue_agent",
		"tactical_agent",
		"scraping_agent",
		"analysis_agent",
		"disaster_analysis_agent",
		"agro_analysis_agent",
		"ecological_analysis_agent",
		"air_quality_analysis_agent",
		"vegetation_analysis_agent",
		"soil_moisture_analysis_agent",
	]

	const handleAgentSelect = (agent: string) => {
		if (selectedAgents.includes(agent)) {
			setSelectedAgents(selectedAgents.filter((a) => a !== agent))
			const updatedSequences = { ...agentSequences }
			delete updatedSequences[agent]
			setAgentSequences(updatedSequences)
		} else {
			setSelectedAgents([...selectedAgents, agent])
			setAgentSequences({
				...agentSequences,
				[agent]: selectedAgents.length + 1,
			})
		}
	}

	const handleSequenceChange = (agent: string, sequence: number) => {
		setAgentSequences({ ...agentSequences, [agent]: sequence })
	}

	const handlerSubmit = async (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault()
		setLoading(true)
		await service
			.post("/create-flow/", {
				flow_name: flowName,
				flow_description: flowDescription,
				agents: selectedAgents,
				agent_sequences: agentSequences,
				flow_type: flowType,
			})
			.then((res) => {
				const { message } = res.data
				if (message) {
					setResponse(message)
				} else {
					console.log("[Error creating flow]", res.data)
					toast.error("Error creating flow")
				}
			})
			.catch((err) => {
				console.log("[Error creating flow]", err)
				toast.error("Error creating flow")
			})
			.finally(() => {
				setLoading(false)
			})
	}

	return (
		<FlowCreateContainer>
			<h6>Create New Flow</h6>
			<Form id="form-flow-test" onSubmit={handlerSubmit}>
				<p>Obligatory fields*</p>
				<div className="row">
					<div className="col-md-6 col-12 mb-3">
						<Form.Group controlId="flowName" className="mb-3">
							<Form.Label>Flow Name*</Form.Label>
							<Form.Control
								required
								type="text"
								value={flowName}
								onChange={(e) => setFlowName(e.target.value)}
								placeholder="Enter flow name"
							/>
						</Form.Group>
					</div>
					<div className="col-md-6 col-12 mb-3">
						<Form.Group controlId="flowType" className="mb-3">
							<Form.Label>Flow Style*</Form.Label>
							<Form.Select
								aria-label="Flow Styled Select"
								required
								value={flowType}
								onChange={(e) => setFlowType(e.target.value)}
							>
								<option value="sequential">Sequential</option>
								<option value="parallel">Parallel</option>
								<option value="adaptive">Adaptive</option>
							</Form.Select>
						</Form.Group>
					</div>
				</div>
				<div className="row">
					<div className="col-md-6 col-12 mb-3">
						<Form.Group controlId="flowName" className="mb-3">
							<Form.Label>Agents*</Form.Label>
							<Dropdown className="dropdown-agents" autoClose="outside">
								<Dropdown.Toggle>
									<p>Select an Agent</p>
									<ChevronDownIcon size={18} />
								</Dropdown.Toggle>
								<Dropdown.Menu>
									<div className="content-input">
										<Form.Control
											type="text"
											placeholder="Search for an agent"
											value={searchValue}
											onChange={(e) => setSearchValue(e.target.value)}
										/>
									</div>
									{availableAgents
										.filter((agent) =>
											agent.toLowerCase().includes(searchValue.toLowerCase()),
										)
										.map((agent) => (
											<Dropdown.Item
												key={agent}
												onClick={() => {
													handleAgentSelect(agent)
												}}
											>
												{agent}
											</Dropdown.Item>
										))}
								</Dropdown.Menu>
							</Dropdown>
						</Form.Group>
					</div>
					{selectedAgents.length !== 0 && (
						<div className="col-md-6 col-12 mb-3 flex-column gap-3 d-flex">
							<Form.Group controlId="agent-sequency" className="mb-3">
								<Form.Label>Agent Sequency</Form.Label>
								{selectedAgents.map((agent: string) => {
									return (
										<InputGroup className="mb-2" key={agent}>
											<InputGroup.Text>{agent}*</InputGroup.Text>
											<Form.Control
												type="number"
												required
												value={agentSequences[agent] || ""}
												onChange={(e) =>
													handleSequenceChange(agent, parseInt(e.target.value))
												}
											/>
											<Button
												variant="danger"
												onClick={() => handleAgentSelect(agent)}
												title="Remove agent"
											>
												<XIcon size={18} />
											</Button>
										</InputGroup>
									)
								})}
							</Form.Group>
						</div>
					)}
				</div>
				<Form.Group controlId="question" className="mb-3">
					<Form.Label>Flow Description*</Form.Label>
					<Form.Control
						required
						as="textarea"
						rows={4}
						value={flowDescription}
						onChange={(e) => setFlowDescription(e.target.value)}
						placeholder="Enter flow description"
					/>
				</Form.Group>

				<div className="action-button">
					<Button type="submit" disabled={loading}>
						{loading ? <Spinner size="sm" color="white" /> : "Test Flow"}
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
		</FlowCreateContainer>
	)
}

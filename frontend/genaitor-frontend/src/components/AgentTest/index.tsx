"use client"
import { Button, Form, Spinner } from "react-bootstrap"
import { ContentAgentTestStyled } from "./styled"
import { FormEvent, useState } from "react"
import LoadingContent from "../LoadingContent"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { service } from "../../../server"
import { toast } from "react-toastify"

export default function AgentTest() {
	const [agent, setAgent] = useState<string>("")
	const [inputData, setInputData] = useState<string>("")
	const [response, setResponse] = useState<string | null>(null)
	const [loading, setLoading] = useState<boolean>(false)

	const agents: string[] = [
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
	].sort((a, b) => a.localeCompare(b))

	const handlerSubmit = async (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault()
		setLoading(true)
		await service
			.post("/process/", {
				agent_name: agent,
				input_data: inputData,
			})
			.then((res) => {
				const { response } = res.data
				setResponse(response.content)
			})
			.catch((err) => {
				console.log("[ERROR processing request]", err)
				toast.error("Error processing request", {
					autoClose: 5000,
				})
			})
			.finally(() => {
				setLoading(false)
			})
	}

	return (
		<ContentAgentTestStyled>
			<h6>Testing Preset Agents</h6>
			<Form id="form-agent-test" onSubmit={handlerSubmit}>
				<p>Obligatory fields*</p>
				<div className="row">
					<div className="col-lg-4 col-md-6 col-12">
						<Form.Group controlId="select-agent">
							<Form.Label>Select an Agent:*</Form.Label>
							<Form.Select
								aria-label="Select an agent"
								value={agent}
								onChange={(e) => {
									setAgent(e.target.value)
								}}
								required
							>
								<option value="">Select one</option>
								{agents.map((ag: string) => {
									return (
										<option key={ag} value={ag}>
											{ag}
										</option>
									)
								})}
							</Form.Select>
						</Form.Group>
					</div>
				</div>
				<Form.Group controlId="data-input">
					<Form.Label>Input Data:*</Form.Label>
					<Form.Control
						required
						as="textarea"
						rows={3}
						value={inputData}
						onChange={(e) => {
							setInputData(e.target.value)
						}}
						placeholder="Enter input data here..."
					/>
				</Form.Group>
				<div className="action-button">
					<Button type="submit" disabled={loading}>
						{loading ? <Spinner size="sm" color="white" /> : "Submit"}
					</Button>
				</div>
			</Form>
			{loading && <LoadingContent />}
			{response && (
				<div className="response-content">
					<h6>Response:</h6>
					<ReactMarkdown remarkPlugins={[remarkGfm]}>{response}</ReactMarkdown>
					{response.includes("```") && (
						<div className="d-flex justify-content-end">
							<Button onClick={() => console.log("Debugging Code:", response)}>
								Debug Code
							</Button>
						</div>
					)}
				</div>
			)}
		</ContentAgentTestStyled>
	)
}

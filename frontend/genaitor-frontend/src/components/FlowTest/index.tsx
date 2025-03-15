"use client"
import { Button, Form, Spinner } from "react-bootstrap"
import { FlowTestContainer } from "./styled"
import { FormEvent, useState } from "react"
import LoadingContent from "../LoadingContent"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { service } from "../../../server"
import { toast } from "react-toastify"

export default function FlowTest() {
	const [question, setQuestion] = useState<string>("")
	const [flowName, setFlowName] = useState<string>("default_flow") // Default flow
	const [response, setResponse] = useState<string | null>(null)
	const [loading, setLoading] = useState<boolean>(false)

	const handlerSubmit = async (e: FormEvent<HTMLFormElement>) => {
		e.preventDefault()
		setLoading(true)
		await service
			.post("/process/", {
				input_data: question,
				flow_name: flowName,
			})
			.then((res) => {
				if (!res.data?.response?.content) {
					toast.error("Invalid response from the server")
					return
				}
				let formattedResponse = ""
				let content

				if (flowName === "default_flow") {
					content = res.data.response.content.gemini
				} else if (flowName === "agent_creation_flow") {
					content = res.data.response.content.creator
				}
				if (content?.success) {
					formattedResponse = content.content.trim()
					formattedResponse = formattedResponse.replace(/\*\*/g, "")
					formattedResponse = formattedResponse
						.split("\n")
						.filter((line) => line.trim())
						.join("\n")
				} else {
					formattedResponse = "Empty response received"
				}
				setResponse(formattedResponse)
			})
			.catch((err) => {
				console.log("[Erro processing request]", err)
				toast.error(err.message || "Error processing request")
			})
			.finally(() => {
				setLoading(false)
			})
	}

	return (
		<FlowTestContainer>
			<h6>Flow Testing</h6>
			<Form id="form-flow-test" onSubmit={handlerSubmit}>
				<p>Obligatory fields*</p>
				<div className="row">
					<div className="col-lg-4 col-md-6 col-12">
						<Form.Group controlId="flowName" className="mb-3">
							<Form.Label>Select Flow*</Form.Label>
							<Form.Select
								required
								value={flowName}
								onChange={(e) => setFlowName(e.target.value)}
							>
								<option value="default_flow">Advanced Usage Flow</option>
								<option value="agent_creation_flow">
									Agent Generator Flow
								</option>
							</Form.Select>
						</Form.Group>
					</div>
				</div>
				<Form.Group controlId="question" className="mb-3">
					<Form.Label>Enter Input*</Form.Label>
					<Form.Control
						required
						as="textarea"
						rows={4}
						value={question}
						onChange={(e) => setQuestion(e.target.value)}
						placeholder="Enter your input here..."
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
		</FlowTestContainer>
	)
}

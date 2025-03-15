"use client"

import { Tab, Tabs } from "react-bootstrap"
import { ContentSection } from "./styled"
import AgentTest from "@/components/AgentTest"
import AgentCreate from "@/components/AgentCreate"
import FlowTest from "@/components/FlowTest"
import FlowCreate from "@/components/FlowCreate"
export default function Index() {
	return (
		<ContentSection>
			<h6>Choose what do you want</h6>
			<div className="overflow-auto">
				<Tabs
					defaultActiveKey="agents"
					id="tabs-content-option-genaitor"
					className="tabs-genaitor"
				>
					<Tab eventKey="agents" title="Use Preset Agents">
						<AgentTest />
					</Tab>
					<Tab eventKey="flows" title="Use Preset Flows">
						<FlowTest />
					</Tab>
					<Tab eventKey="create-flow" title="Create new Flow">
						<FlowCreate />
					</Tab>
					<Tab eventKey="create-agent" title="Create New Task and Agent">
						<AgentCreate />
					</Tab>
				</Tabs>
			</div>
		</ContentSection>
	)
}

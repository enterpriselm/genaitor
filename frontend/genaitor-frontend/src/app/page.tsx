"use client"

import { Tab, Tabs } from "react-bootstrap"
import { ContentSection } from "./styled"
import AgentTest from "@/components/AgentTest"
import AgentCreate from "@/components/AgentCreate"
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
					<Tab eventKey="agents" title="Test Preset Agents">
						<AgentTest />
					</Tab>
					<Tab eventKey="create-agent" title="Create New Task and Agent">
						<AgentCreate />
					</Tab>
					<Tab eventKey="create-flow" title="Create your Flow">
						<FlowCreate />
					</Tab>
					
				</Tabs>
			</div>
		</ContentSection>
	)
}

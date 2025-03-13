"use client"

import { Tab, Tabs } from "react-bootstrap"
import { ContentSection } from "./styled"

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
					<Tab eventKey="agents" title="Use Preset Agents"></Tab>
					<Tab eventKey="flows" title="Use Preset Flows"></Tab>
					<Tab eventKey="create-flow" title="Create new Flow"></Tab>
					<Tab eventKey="create-agent" title="Create New Task and Agent"></Tab>
				</Tabs>
			</div>
		</ContentSection>
	)
}

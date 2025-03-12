// src/components/FlowCreateScreen.tsx
import React, { useState } from "react";
import axios from "axios";

interface FlowCreateScreenProps {}

const FlowCreateScreen: React.FC<FlowCreateScreenProps> = () => {
  const [flowName, setFlowName] = useState<string>("");
  const [flowDescription, setFlowDescription] = useState<string>("");
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [agentSequences, setAgentSequences] = useState<{ [key: string]: number }>({});
  const [flowType, setFlowType] = useState<string>("sequential");
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Lista de agentes (substitua com a sua lista real de agentes)
  const availableAgents = [
    "qa_agent", "autism_agent", "agent_creation", "data_understanding_agent", "statistics_agent",
    "anomalies_detection_agent", "data_analysis_agent", "problem_analysis_agent", "numerical_analysis_agent",
    "pinn_modeling_agent", "preferences_agent", "payment_agent", "proposal_agent",
    "review_agent", "extraction_agent", "matching_agent", "scoring_agent", "report_agent",
    "optimization_agent", "educational_agent", "research_agent", "content_agent", "personalization_agent",
    "financial_agent", "summatization_agent", "linkedin_agent", "pinn_tuning_agent", "html_analysis_agent",
    "scraper_generation_agent", "equation_solver_agent", "pinn_generation_agent", "hyperparameter_optimization_agent",
    "orchestrator_agent", "validator_agent", "requirements_agent", "architecture_agent", "code_generation_agent",
    "destination_selection_agent", "budget_estimation_agent", "itinerary_planning_agent", "feature_selection_agent",
    "signal_analysis_agent", "residual_evaluation_agent", "lstm_model_agent", "lstm_residual_evaluation_agent",
    "document_agent", "question_agent", "search_agent", "response_agent", "performance_agent", "fatigue_agent",
    "tactical_agent", "scraping_agent", "analysis_agent", "disaster_analysis_agent", "agro_analysis_agent",
    "ecological_analysis_agent", "air_quality_analysis_agent", "vegetation_analysis_agent", "soil_moisture_analysis_agent"
  ];

  const handleAgentSelect = (agent: string) => {
    if (selectedAgents.includes(agent)) {
      setSelectedAgents(selectedAgents.filter((a) => a !== agent));
      const updatedSequences = { ...agentSequences };
      delete updatedSequences[agent];
      setAgentSequences(updatedSequences);
    } else {
      setSelectedAgents([...selectedAgents, agent]);
      setAgentSequences({ ...agentSequences, [agent]: selectedAgents.length + 1 });
    }
  };

  const handleSequenceChange = (agent: string, sequence: number) => {
    setAgentSequences({ ...agentSequences, [agent]: sequence });
  };

  const handleCreateFlow = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await axios.post("http://127.0.0.1:8000/create-flow", {
        flow_name: flowName,
        flow_description: flowDescription,
        agents: selectedAgents,
        agent_sequences: agentSequences,
        flow_type: flowType,
      });

      if (res.data && res.data.message) {
        setResponse(res.data.message);
      } else {
        setError("Error creating flow");
      }
    } catch (err: any) {
      setError(err.message || "Error creating flow");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 flex items-center justify-center">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-semibold mb-4 text-center">
          Create New Flow
        </h1>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Flow Name:
          </label>
          <br></br>
          <input
            type="text"
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={flowName}
            onChange={(e) => setFlowName(e.target.value)}
            placeholder="Enter flow name"
          />
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Flow Description:
          </label>
          <br></br>
          <textarea
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            rows={4}
            value={flowDescription}
            onChange={(e) => setFlowDescription(e.target.value)}
            placeholder="Enter flow description"
          />
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Agents:
          </label>
          <br></br>
          <select
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={selectedAgents}
            onChange={(e) => handleAgentSelect(e.target.value)}
          >
            <option value="">Select an Agent</option>
            {availableAgents.map((agent) => (
              <option key={agent} value={agent}>
                {agent}
              </option>
            ))}
          </select>
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Agent Sequency:
          </label>
          <br></br>
          {selectedAgents.map((agent) => (
            <div key={agent} className="flex items-center space-x-2">
              <span className="text-sm">{agent}:</span>
              <input
                type="number"
                className="mt-1 block w-20 rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
                value={agentSequences[agent] || ""}
                onChange={(e) => handleSequenceChange(agent, parseInt(e.target.value))}
              />
            </div>
          ))}
        </div>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Flow Style:
          </label>
          <select
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={flowType}
            onChange={(e) => setFlowType(e.target.value)}
          >
            <option value="sequential">Sequential</option>
            <option value="parallel">Parallel</option>
            <option value="adaptive">Adaptive</option>
          </select>
        </div>

        <button
          className="w-full py-2 px-4 bg-green-600 hover:bg-green-700 rounded-md text-white font-semibold"
          onClick={handleCreateFlow}
          disabled={loading}
        >
          {loading ? "Creating..." :"Create Flow"}
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

    export default FlowCreateScreen;
    
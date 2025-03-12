// src/components/AgentTestScreen.tsx
import React, { useState, useEffect } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface AgentTestScreenProps {}

const AgentTestScreen: React.FC<AgentTestScreenProps> = () => {
  const [agent, setAgent] = useState<string>("");
  const [inputData, setInputData] = useState<string>("");
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const agents: string[] = [
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

  const sortedAgents = [...agents].sort();

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.post("http://127.0.0.1:8000/process/", {
        agent_name: agent,
        input_data: inputData,
      });
      setResponse(res.data.response.content);
    } catch (err: any) {
      setError("Error processing request");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-semibold mb-4 text-center">
          Testing Preset Agents
        </h1>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Select an Agent:
          </label>
          <br></br>
          <select
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
          >
            <option value="">Choose an agent</option>
            {sortedAgents.map((ag) => (
              <option key={ag} value={ag}>
                {ag}
              </option>
            ))}
          </select>
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Input Data:
          </label>
          <br></br>
          <textarea
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            rows={4}
            value={inputData}
            onChange={(e) => setInputData(e.target.value)}
            placeholder="Enter input data here..."
          />
        </div>

        <button
          className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 rounded-md text-white font-semibold"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Processing..." : "Submit"}
        </button>

        {error && <p className="text-red-500 mt-2 text-center">{error}</p>}

        {response && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h2 className="text-lg font-semibold mb-2">Response:</h2>
            <ReactMarkdown remarkPlugins={[remarkGfm]} children={response} />
          </div>
        )}
      </div>
    </div>
  );
};

export default AgentTestScreen;
// src/components/FlowTestScreen.tsx
import React, { useState } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface FlowTestScreenProps {}

const FlowTestScreen: React.FC<FlowTestScreenProps> = () => {
  const [question, setQuestion] = useState<string>("");
  const [flowName, setFlowName] = useState<string>("default_flow"); // Default flow
  const [response, setResponse] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleTestFlow = async () => {
    setLoading(true);
    setError(null);
    setResponse(null);

    try {
      const res = await axios.post("http://127.0.0.1:8000/process/", {
        input_data: question,
        flow_name: flowName,
      });

      if (res.data && res.data.response && res.data.response.content) {
        let formattedResponse = "";
        let content;

        if (flowName === "default_flow") {
          content = res.data.response.content.gemini; // Advanced Usage
        } else if (flowName === "agent_creation_flow") {
          content = res.data.response.content.creator; // Agent Generator
        }

        if (content && content.success) {
          formattedResponse = content.content.trim();
          formattedResponse = formattedResponse.replace(/\*\*/g, "");
          formattedResponse = formattedResponse
            .split("\n")
            .filter((line) => line.trim())
            .join("\n");
        } else {
          formattedResponse = "Empty response received";
        }
        setResponse(formattedResponse);
      } else {
        setError("Invalid response from the server");
      }
    } catch (err: any) {
      setError(err.message || "Error processing request");
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-100 p-4 flex items-center justify-center">
      <div className="max-w-3xl mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-semibold mb-4 text-center">
          Flow Testing
        </h1>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Select Flow:
          </label>
          <br></br>
          <select
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            value={flowName}
            onChange={(e) => setFlowName(e.target.value)}
          >
            <option value="default_flow">Advanced Usage Flow</option>
            <option value="agent_creation_flow">Agent Generator Flow</option>
          </select>
        </div>
        <br></br>
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Enter Input:
          </label>
          <br></br>
          <textarea
            className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-300 focus:ring focus:ring-indigo-200 focus:ring-opacity-50"
            rows={4}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your input here..."
          />
        </div>

        <button
          className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 rounded-md text-white font-semibold"
          onClick={handleTestFlow}
          disabled={loading}
        >
          {loading ? "Testing..." : "Test Flow"}
        </button>

        {error && <p className="text-red-500 mt-2 text-center">{error}</p>}

        {response && (
          <div className="mt-6 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h2 className="text-lg font-semibold mb-2">Response:</h2>
            <ReactMarkdown remarkPlugins={[remarkGfm]} children={response} />
            
            {response.includes("```") && (
              <button
                className="mt-2 py-2 px-4 bg-green-600 hover:bg-green-700 rounded-md text-white font-semibold"
                onClick={() => console.log("Debugging Code:", response)}
              >
                Debug Code
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default FlowTestScreen;
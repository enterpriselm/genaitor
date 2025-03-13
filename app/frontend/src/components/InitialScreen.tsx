import React from "react";
import { Link } from "react-router-dom";

function InitialScreen() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <h1 className="text-3xl font-bold mb-4">GenAltor</h1>
        <div>
          <Link to="/agents" className="text-blue-500 hover:underline mr-4">
            Use Preseted Agents
          </Link>
          <Link to="/flows" className="text-blue-500 hover:underline mr-4">
            Use Preseted Flows
          </Link>
          <Link to="/flows/create" className="text-blue-500 hover:underline mr-4">
            Create new Flow
          </Link>
          <Link to="/agents/create" className="text-blue-500 hover:underline">
            Create New Task and Agent
          </Link>
        </div>
      </div>
    </div>
  );
}

export default InitialScreen;
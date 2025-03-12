// src/components/InitialScreen.tsx
import React from "react";
import { Link } from "react-router-dom";

function InitialScreen() {
  return (
    <div className="min-h-screen bg-gray-100 p-4 flex items-center justify-center">
      <div className="max-w-md mx-auto bg-white rounded-lg shadow-md p-6">
        <h1 className="text-2xl font-semibold mb-6 text-center">
          GenAItor
        </h1>
        <div className="grid grid-cols-1 gap-4">
          <Link
            to="/agents"
            className="py-2 px-4 bg-indigo-600 hover:bg-indigo-700 rounded-md text-white font-semibold text-center"
          >
            Use Preseted Agents
          </Link>
          <br></br>
          <Link
            to="/flows"
            className="py-2 px-4 bg-indigo-600 hover:bg-indigo-700 rounded-md text-white font-semibold text-center"
          >
            Use Preseted Flows
          </Link>
          <br></br>
          <Link
            to="/flows/create"
            className="py-2 px-4 bg-green-600 hover:bg-green-700 rounded-md text-white font-semibold text-center"
          >
            Create new Flow
          </Link>
          <br></br>
          <Link
            to="/agents/create"
            className="py-2 px-4 bg-green-600 hover:bg-green-700 rounded-md text-white font-semibold text-center"
          >
            Create New Task and Agent
          </Link>
        </div>
      </div>
    </div>
  );
}

export default InitialScreen;
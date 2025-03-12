// src/App.tsx
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import InitialScreen from "./components/InitialScreen";
import AgentTestScreen from "./components/AgentTestScreen";
import FlowTestScreen from "./components/FlowTestScreen";
import FlowCreateScreen from "./components/FlowCreateScreen";
import AgentCreateScreen from "./components/AgentCreateScreen";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<InitialScreen />} />
        <Route path="/agents" element={<AgentTestScreen />} />
        <Route path="/flows" element={<FlowTestScreen />} />
        <Route path="/flows/create" element={<FlowCreateScreen />} />
        <Route path="/agents/create" element={<AgentCreateScreen />} />
      </Routes>
    </Router>
  );
}

export default App;
import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // You can style your app here

const App = () => {
  const [userQuery, setUserQuery] = useState('');
  const [aiResponse, setAiResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    
    try {
      const response = await axios.post('http://localhost:5000/generate-agent', {
        user_query: userQuery
      }//, {
       // headers: {
       //   'X-API-Key': 'your-api-key-here'
       // }
      //}
    );

      setAiResponse(response.data.ai_agent_prompt);
    } catch (err) {
      setError(err.response?.data?.error || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <h1>AI Agent Generator</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={userQuery}
          onChange={(e) => setUserQuery(e.target.value)}
          placeholder="Enter your prompt here..."
          required
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Generating...' : 'Generate Agent'}
        </button>
      </form>
      {error && <div className="error">{error}</div>}
      {aiResponse && (
        <div className="response">
          <h2>AI Response:</h2>
          <p>{aiResponse}</p>
        </div>
      )}
    </div>
  );
};

export default App;

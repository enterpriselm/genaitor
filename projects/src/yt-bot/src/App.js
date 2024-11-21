import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [userQuery, setUserQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResponse(''); // Limpa a resposta anterior antes de uma nova chamada

    try {
      const result = await axios.post(
        'http://localhost:5000/youtube',
        { youtube_url: youtubeUrl, user_query: userQuery }, // Payload em JSON
        {
          headers: {
            'Content-Type': 'application/json',
            // 'X-API-Key': 'your-api-key-here', // Descomente se necessário
          },
        }
      );

      if (result.data && result.data.answer) {
        setResponse(result.data.answer);
      } else {
        setResponse('No answer received from the API.');
      }
    } catch (error) {
      console.error(error);
      setResponse('Error processing request. Check console for details.');
    } finally {
      setLoading(false); // Move o setLoading para o bloco finally para garantir que seja chamado em caso de erro
    }
  };

  return (
    <div className="App">
      <h1>YT Video AI Helper</h1>
      <form onSubmit={handleFormSubmit}>
        <label> YouTube URL:
          <input
            type="text"
            value={youtubeUrl}
            onChange={(e) => setYoutubeUrl(e.target.value)}
            placeholder="Enter YouTube URL"
          />
        </label>
        
        <label> User Query:
          <textarea
            value={userQuery}
            onChange={(e) => setUserQuery(e.target.value)}
            placeholder="Enter your question about the video"
          />
        </label>
        
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Get Answer'}
        </button>
      </form>

      <div className="response">
        <h2>AI Response:</h2>
        <p>{response}</p>
      </div>
    </div>
  );
}

export default App;

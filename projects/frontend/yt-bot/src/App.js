import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [userQuery, setUserQuery] = useState('');
  const [videoFile, setVideoFile] = useState(null);
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFormSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    const formData = new FormData();
    formData.append('user_query', userQuery);
    if (youtubeUrl) formData.append('youtube_url', youtubeUrl);
    if (videoFile) formData.append('video_file', videoFile);
    
    console.log(formData)
    try {
      const result = await axios.post('http://localhost:5000/youtube', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          // 'X-API-Key': 'your-api-key-here',
        },
      });
      setResponse(result.data);
    } catch (error) {
      console.error(error);
      setResponse('Error processing request. Check console for details.');
    }
    setLoading(false);
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
        
        <label> Upload Video File (.mp4/.mp3):
          <input
            type="file"
            accept=".mp4, .mp3"
            onChange={(e) => setVideoFile(e.target.files[0])}
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

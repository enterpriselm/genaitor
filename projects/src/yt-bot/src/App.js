import React, { useState } from "react";

const App = () => {
    const [files, setFiles] = useState([]);
    const [paths, setPaths] = useState([]);
    const [query, setQuery] = useState('');
    const [apiResponse, setApiResponse] = useState('');

    const handleFileChange = (e) => {
        const selectedFiles = Array.from(e.target.files);
        setFiles(selectedFiles);
    };

    const handleQueryChange = (e) => {
      setQuery(e.target.value);
    }

    const isFileSecure = (file) => {
        const allowedExtensions = [
            "pdf", "doc", "docx", "json", "ppt", "pptx",
            "xls", "xlsx", "csv", "jpg", "jpeg", "png", "mp3", "mp4",
        ];
        const extension = file.name.split(".").pop().toLowerCase();
        return allowedExtensions.includes(extension);
    };

    const downloadFiles = () => {
        const validPaths = [];
        files.forEach((file) => {
            if (isFileSecure(file)) {
                const blob = new Blob([file], { type: file.type });
                const url = URL.createObjectURL(blob);
                const link = document.createElement("a");
                link.href = url;
                link.download = file.name;
                link.click();
                URL.revokeObjectURL(url);
                validPaths.push('/home/usuario/Downloads/'+file.name); // Simulated path
            } else {
                alert(`${file.name} is not secure and will not be downloaded.`);
            }
        });
        setPaths(validPaths);
    };

    const submitPaths = async () => {
        if (paths.length === 0) {
            alert("No valid files to submit.");
            return;
        }
        try {
            const response = await fetch("http://localhost:5000/text_analyzer", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    media_data: paths,
                    user_query: query,
                }),
            });
            const data = await response.json();
            setApiResponse(JSON.stringify(data, null, 2));
        } catch (error) {
            console.error("Error submitting paths:", error);
            setApiResponse("Error connecting to the API.")
        }
    };

    return (
        <div style={{ padding: "20px" }}>
            <h1>File Upload Interface</h1>

            <input
                type="text"
                placeholder="Enter your query"
                value={query}
                onChange={handleQueryChange}
                style={{ display: "block", marginBottom: "10px", width: "100%" }}
            />

            <input
                type="file"
                multiple
                onChange={handleFileChange}
                style={{ display: "block", marginBottom: "20px" }}
            />

            <button
                onClick={downloadFiles}
                style={{ display: "block", marginBottom: "10px", width: "100%" }}
            >
                Check & Download
            </button>
            <button
                onClick={submitPaths}
                style={{ display: "block", width: "100%" }}
            >
                Submit to API
            </button>

            <div style={{ marginTop: "20px" }}>
                <h3>Uploaded Files:</h3>
                <ul>
                    {files.map((file, index) => (
                        <li key={index}>{file.name}</li>
                    ))}
                </ul>
            </div>

            <div style={{ marginTop: "20px" }}>
                <h3>Valid Paths:</h3>
                <ul>
                    {paths.map((path, index) => (
                        <li key={index}>{path}</li>
                    ))}
                </ul>
            </div>

            {/* Resposta da API */}
            <div style={{ marginTop: "20px" }}>
                <h3>API Response:</h3>
                <p style={{ whiteSpace: "pre-wrap", backgroundColor: "#f0f0f0", padding: "10px", borderRadius: "5px" }}>
                    {apiResponse || "No response yet."}
                </p>
            </div>
        </div>
    );
};

export default App;
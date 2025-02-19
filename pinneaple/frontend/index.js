import React from 'react';
import ReactDOM from 'react-dom/client'; // Certifique-se de que está importando de 'react-dom/client'
import ProviderSelector from './components/ProviderSelector';
import './index.css'; // Se você tiver um CSS global

const root = ReactDOM.createRoot(document.getElementById('root')); // Certifique-se de que o ID 'root' está correto
root.render(
    <React.StrictMode>
        <ProviderSelector />
    </React.StrictMode>
); 
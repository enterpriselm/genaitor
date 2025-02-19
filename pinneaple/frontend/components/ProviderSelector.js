import React, { useState } from 'react';
import styles from './ProviderSelector.module.css';

const providersData = {
    anthropic: { models: ['claude-3-opus-20240229'], apiFields: [{ label: 'API Key', id: 'api_key', placeholder: 'Enter your API Key' }] },
    deepseek: { apiFields: [{ label: 'API Key', id: 'api_key', placeholder: 'Enter your API Key' }, { label: 'Base URL', id: 'base_url', placeholder: 'Enter the Base URL' }] },
    custom: { apiFields: [{ label: 'API URL', id: 'custom_url', placeholder: 'Enter the API URL' }] },
    google: { models: ['gemini-pro'], apiFields: [{ label: 'API Key', id: 'api_key', placeholder: 'Enter your API Key' }] },
    ollama: { models: ['llama3.2'], apiFields: [{ label: 'Base URL', id: 'base_url', placeholder: 'Enter the Base URL' }] },
    openai: { models: ['gpt-4'], apiFields: [{ label: 'API Key', id: 'api_key', placeholder: 'Enter your API Key' }, { label: 'Organization (optional)', id: 'organization', placeholder: 'Enter your Organization' }] },
};

const ProviderSelector = () => {
    const [selectedProvider, setSelectedProvider] = useState('');
    const [selectedModel, setSelectedModel] = useState('');

    const handleProviderChange = (e) => {
        setSelectedProvider(e.target.value);
        setSelectedModel('');
    };

    const providerData = providersData[selectedProvider] || {};

    return (
        <div className={styles.container}>
            <label htmlFor="provider-select">Select Provider:</label>
            <select id="provider-select" onChange={handleProviderChange}>
                <option value="">--Select--</option>
                {Object.keys(providersData).map(provider => (
                    <option key={provider} value={provider}>{provider.charAt(0).toUpperCase() + provider.slice(1)}</option>
                ))}
            </select>

            {providerData.models && providerData.models.length > 0 && selectedProvider !== 'deepseek' && selectedProvider !== 'custom' && (
                <>
                    <label htmlFor="model-select">Model:</label>
                    <select id="model-select" onChange={(e) => setSelectedModel(e.target.value)}>
                        <option value="">--Select--</option>
                        {providerData.models.map(model => (
                            <option key={model} value={model}>{model}</option>
                        ))}
                    </select>
                </>
            )}

            {providerData.apiFields && providerData.apiFields.map(field => (
                <div key={field.id} className={styles.formGroup}>
                    <label htmlFor={field.id}>{field.label}:</label>
                    <input type="text" id={field.id} placeholder={field.placeholder} />
                </div>
            ))}
        </div>
    );
};

export default ProviderSelector; 
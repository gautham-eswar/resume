import { useState } from 'react';

function DebugView({ responses = {} }) {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('parsing');
  
  const tabs = [
    { id: 'parsing', label: 'Resume Parsing' },
    { id: 'keywords', label: 'Keyword Extraction' },
    { id: 'matching', label: 'Matching' },
    { id: 'enhancement', label: 'Enhancement' }
  ];
  
  // Only show tabs for which we have data
  const availableTabs = tabs.filter(tab => responses[tab.id]);

  if (availableTabs.length === 0) {
    return null;
  }

  return (
    <div className="debug-view">
      <button 
        className="toggle-btn"
        onClick={() => setIsOpen(!isOpen)}
      >
        {isOpen ? 'Hide' : 'Show'} API Responses
      </button>
      
      {isOpen && (
        <div className="debug-content">
          <div className="debug-tabs">
            {availableTabs.map(tab => (
              <button
                key={tab.id}
                className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          
          <div className="debug-panel">
            <pre className="json-view">
              {responses[activeTab] ? 
                JSON.stringify(responses[activeTab], null, 2) : 
                'No data available'
              }
            </pre>
          </div>
        </div>
      )}
    </div>
  );
}

export default DebugView; 
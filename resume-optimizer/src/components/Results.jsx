import { useState } from 'react';
import { getDownloadUrl } from '../services/api';

function Results({ modifications = [], resumeId, statistics }) {
  const [showModifiedOnly, setShowModifiedOnly] = useState(true);

  if (!modifications.length) {
    return null;
  }

  // Group modifications by company/experience
  const groupedModifications = modifications.reduce((acc, mod) => {
    const key = `${mod.company} - ${mod.position}`;
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(mod);
    return acc;
  }, {});

  const handleDownload = (format) => {
    if (!resumeId) return;
    
    // Create an anchor element and trigger download
    const link = document.createElement('a');
    link.href = getDownloadUrl(resumeId, format);
    link.download = `enhanced-resume.${format}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const highlightKeywords = (text, keywords) => {
    if (!keywords || !keywords.length) return text;
    
    let highlightedText = text;
    keywords.forEach(keyword => {
      // Simple case-insensitive replacement - in a real app, you might want more sophisticated matching
      const regex = new RegExp(`(${keyword})`, 'gi');
      highlightedText = highlightedText.replace(regex, '<span class="highlight">$1</span>');
    });
    
    return <div dangerouslySetInnerHTML={{ __html: highlightedText }} />;
  };

  return (
    <div className="results-container">
      <h2>Enhancement Results</h2>
      
      {statistics && (
        <div className="statistics">
          <h3>Summary</h3>
          <ul>
            <li>Keywords extracted: {statistics.keywords_extracted}</li>
            <li>Keywords matched in resume: {statistics.keywords_deduplicated}</li>
            <li>Bullets enhanced: {statistics.bullets_enhanced} of {statistics.bullets_processed}</li>
          </ul>
        </div>
      )}
      
      <div className="download-buttons">
        <button 
          onClick={() => handleDownload('json')}
          className="download-btn json"
          disabled={!resumeId}
        >
          Download Enhanced JSON
        </button>
        <button 
          onClick={() => handleDownload('pdf')}
          className="download-btn pdf"
          disabled={!resumeId}
        >
          Download Enhanced PDF
        </button>
      </div>
      
      <div className="results-controls">
        <label>
          <input
            type="checkbox"
            checked={showModifiedOnly}
            onChange={(e) => setShowModifiedOnly(e.target.checked)}
          />
          Show only modified bullets
        </label>
      </div>

      <div className="comparison-view">
        <div className="header">
          <div className="original">Original</div>
          <div className="enhanced">Enhanced</div>
        </div>
        
        {Object.entries(groupedModifications).map(([group, mods]) => {
          // Filter if showing only modified bullets
          const filteredMods = showModifiedOnly ? 
            mods.filter(mod => mod.original_bullet !== mod.enhanced_bullet) : 
            mods;
            
          if (filteredMods.length === 0) return null;
          
          return (
            <div key={group} className="experience-group">
              <h3 className="group-header">{group}</h3>
              
              {filteredMods.map((mod, idx) => (
                <div key={idx} className="bullet-comparison">
                  <div className="original-bullet">
                    {mod.original_bullet}
                  </div>
                  <div className="enhanced-bullet">
                    {highlightKeywords(mod.enhanced_bullet, mod.keywords_added)}
                  </div>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default Results; 
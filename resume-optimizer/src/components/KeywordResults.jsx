import { useState } from 'react';

function KeywordResults({ keywords = [] }) {
  const [sortBy, setSortBy] = useState('relevance');
  const [showMissingOnly, setShowMissingOnly] = useState(false);
  
  // Filter keywords
  const filteredKeywords = showMissingOnly 
    ? keywords.filter(keyword => !keyword.matched) 
    : keywords;
  
  // Sort keywords
  const sortedKeywords = [...filteredKeywords].sort((a, b) => {
    if (sortBy === 'relevance') {
      return b.relevance_score - a.relevance_score;
    } else if (sortBy === 'type') {
      return a.skill_type.localeCompare(b.skill_type);
    } else if (sortBy === 'matched') {
      // Sort by matched status (matched first)
      return (b.matched ? 1 : 0) - (a.matched ? 1 : 0);
    }
    return 0;
  });

  const handleSortChange = (e) => {
    setSortBy(e.target.value);
  };

  const handleShowMissingOnly = (e) => {
    setShowMissingOnly(e.target.checked);
  };

  if (!keywords.length) {
    return null;
  }

  return (
    <div className="keyword-results">
      <h2>Extracted Keywords</h2>
      <div className="controls">
        <select onChange={handleSortChange} value={sortBy}>
          <option value="relevance">Sort by Relevance</option>
          <option value="type">Sort by Type</option>
          <option value="matched">Sort by Match Status</option>
        </select>
        <label>
          <input 
            type="checkbox" 
            checked={showMissingOnly}
            onChange={handleShowMissingOnly} 
          />
          Show Missing Keywords Only
        </label>
      </div>
      
      <table className="keyword-table">
        <thead>
          <tr>
            <th>Keyword</th>
            <th>Type</th>
            <th>Relevance</th>
            <th>Context</th>
            <th>Status</th>
            <th>Similarity</th>
          </tr>
        </thead>
        <tbody>
          {sortedKeywords.map((keyword, index) => (
            <tr 
              key={`${keyword.keyword}-${index}`} 
              className={keyword.selected ? 'selected' : ''}
            >
              <td>{keyword.keyword}</td>
              <td>{keyword.skill_type}</td>
              <td>{keyword.relevance_score}/10</td>
              <td>{keyword.context}</td>
              <td className={keyword.matched ? 'matched' : 'missing'}>
                {keyword.matched ? 'Found' : 'Missing'}
              </td>
              <td>{keyword.similarity ? keyword.similarity.toFixed(2) : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default KeywordResults; 
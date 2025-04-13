import { useState } from 'react';

function JobDescription({ onSubmit, isResumeUploaded, isProcessing }) {
  const [jobDescription, setJobDescription] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (jobDescription.trim() && isResumeUploaded && !isProcessing) {
      onSubmit(jobDescription);
    }
  };

  return (
    <div className="job-description-container">
      <h2>Job Description</h2>
      <form onSubmit={handleSubmit}>
        <textarea
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          placeholder="Paste the job description here..."
          disabled={!isResumeUploaded || isProcessing}
          rows={10}
          className="job-description-textarea"
        />
        <button 
          type="submit" 
          disabled={!jobDescription.trim() || !isResumeUploaded || isProcessing}
          className="analyze-btn"
        >
          {isProcessing ? 'Processing...' : 'Analyze Resume'}
        </button>
      </form>
    </div>
  );
}

export default JobDescription; 
function ProcessSteps({ currentStep, stats }) {
  const steps = [
    { id: 'parsing', name: 'Parsing Resume', description: 'Extracting text and structure from your resume' },
    { id: 'keywords', name: 'Extracting Keywords', description: 'Identifying important keywords from job description' },
    { id: 'matching', name: 'Matching & Analysis', description: 'Analyzing resume against job requirements' },
    { id: 'enhancement', name: 'Resume Enhancement', description: 'Optimizing resume content' }
  ];

  return (
    <div className="process-steps-container">
      <h2>Process Progress</h2>
      <div className="steps">
        {steps.map((step, index) => {
          let status = 'pending';
          if (index < steps.findIndex(s => s.id === currentStep)) {
            status = 'complete';
          } else if (step.id === currentStep) {
            status = 'in-progress';
          }

          return (
            <div key={step.id} className={`step ${status}`}>
              <div className="step-indicator">
                {status === 'complete' ? '✓' : index + 1}
              </div>
              <div className="step-content">
                <h3>{step.name}</h3>
                <p>{step.description}</p>
                {status === 'complete' && stats && stats[step.id] && (
                  <div className="step-stats">
                    {step.id === 'parsing' && (
                      <p>Parsed {stats.parsing.sections} sections, {stats.parsing.bullets} bullets</p>
                    )}
                    {step.id === 'keywords' && (
                      <p>Extracted {stats.keywords.total} keywords ({stats.keywords.hardSkills} hard, {stats.keywords.softSkills} soft)</p>
                    )}
                    {step.id === 'matching' && (
                      <p>Found {stats.matching.matched} matches out of {stats.matching.total} keywords</p>
                    )}
                    {step.id === 'enhancement' && (
                      <p>Enhanced {stats.enhancement.enhanced} out of {stats.enhancement.total} bullets</p>
                    )}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

export default ProcessSteps; 
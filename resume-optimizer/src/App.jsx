import { useState, useEffect } from 'react'
import './App.css'
import FileUpload from './components/FileUpload.jsx'
import JobDescription from './components/JobDescription.jsx'
import ProcessSteps from './components/ProcessSteps.jsx'
import KeywordResults from './components/KeywordResults.jsx'
import Results from './components/Results.jsx'
import DebugView from './components/DebugView.jsx'
import { uploadResume, optimizeResume, testApiConnection } from './services/api'

function App() {
  // API connectivity state
  const [apiConnected, setApiConnected] = useState(null);
  
  // State for uploaded resume
  const [resumeId, setResumeId] = useState(null)
  const [parsedResume, setParsedResume] = useState(null)
  
  // State for optimization process
  const [currentStep, setCurrentStep] = useState(null)
  const [isProcessing, setIsProcessing] = useState(false)
  
  // State for results
  const [keywords, setKeywords] = useState([])
  const [modifications, setModifications] = useState([])
  const [statistics, setStatistics] = useState(null)
  
  // Debug state for API responses
  const [apiResponses, setApiResponses] = useState({})
  
  // Error state
  const [error, setError] = useState(null)
  
  // Stats for process steps
  const [processStats, setProcessStats] = useState({
    parsing: null,
    keywords: null,
    matching: null,
    enhancement: null
  })
  
  // Test API connectivity on component mount
  useEffect(() => {
    const checkApiConnection = async () => {
      try {
        setError(null);
        const result = await testApiConnection();
        if (result.status === 'success') {
          setApiConnected(true);
          console.log("API connection successful:", result);
        } else {
          setApiConnected(false);
          setError("API responded but returned an unexpected status");
        }
      } catch (error) {
        console.error("API connection failed:", error);
        setApiConnected(false);
        setError(`API connection failed: ${error.message}`);
      }
    };
    
    checkApiConnection();
  }, []);

  // Handle resume file upload
  const handleFileUpload = async (file) => {
    try {
      setError(null)
      setCurrentStep('parsing')
      setIsProcessing(true)
      
      console.log("Starting file upload to backend...")
      const response = await uploadResume(file)
      console.log("Backend response:", response)
      
      // Store the response
      setResumeId(response.resumeId)
      setParsedResume(response.parsedResume)
      
      // Update stats
      setProcessStats(prev => ({
        ...prev,
        parsing: {
          sections: response.parsedResume?.sections?.length || 0,
          bullets: countBullets(response.parsedResume)
        }
      }))
      
      // Store for debug view
      setApiResponses(prev => ({
        ...prev,
        parsing: response
      }))
      
      setCurrentStep(null) // Reset current step
    } catch (error) {
      console.error('Error uploading resume:', error)
      setError(`Upload failed: ${error.message}`)
    } finally {
      setIsProcessing(false)
    }
  }

  // Count bullets in parsed resume (helper function)
  const countBullets = (resume) => {
    if (!resume?.experiences) return 0
    
    return resume.experiences.reduce((total, exp) => {
      return total + (exp.bullets?.length || 0)
    }, 0)
  }

  // Handle job description submission and resume optimization
  const handleOptimize = async (jobDescription) => {
    if (!resumeId) return
    
    try {
      setIsProcessing(true)
      
      // Step 1: Extract Keywords
      setCurrentStep('keywords')
      const response = await optimizeResume(resumeId, jobDescription)
      
      // Process response data
      // Store keywords with added 'matched' flag
      const keywordsWithMatches = response.keywords.map(keyword => ({
        ...keyword,
        matched: response.matchDetails.some(
          match => match.keyword === keyword.keyword
        ),
        similarity: response.matchDetails.find(
          match => match.keyword === keyword.keyword
        )?.similarity || 0
      }))
      
      setKeywords(keywordsWithMatches)
      
      // Update keyword stats
      setProcessStats(prev => ({
        ...prev,
        keywords: {
          total: response.keywordsExtracted,
          hardSkills: keywordsWithMatches.filter(k => k.skill_type === 'hard skill').length,
          softSkills: keywordsWithMatches.filter(k => k.skill_type === 'soft skill').length
        }
      }))
      
      // Step 2: Matching & Analysis
      setCurrentStep('matching')
      
      // Update matching stats
      setProcessStats(prev => ({
        ...prev,
        matching: {
          matched: keywordsWithMatches.filter(k => k.matched).length,
          total: keywordsWithMatches.length
        }
      }))
      
      // Step 3: Enhancement
      setCurrentStep('enhancement')
      
      // Store modifications
      setModifications(response.modifications)
      
      // Set statistics
      setStatistics(response.statistics || {
        keywords_extracted: response.keywordsExtracted,
        keywords_deduplicated: keywordsWithMatches.length,
        bullets_processed: countBullets(parsedResume),
        bullets_enhanced: response.modifications.length
      })
      
      // Update enhancement stats
      setProcessStats(prev => ({
        ...prev,
        enhancement: {
          enhanced: response.modifications.filter(
            m => m.original_bullet !== m.enhanced_bullet
          ).length,
          total: response.modifications.length
        }
      }))
      
      // Store for debug view
      setApiResponses(prev => ({
        ...prev,
        keywords: { keywords: keywordsWithMatches },
        matching: { matches: response.matchDetails },
        enhancement: { modifications: response.modifications }
      }))
      
      setCurrentStep(null) // Reset current step
    } catch (error) {
      console.error('Error optimizing resume:', error)
    } finally {
      setIsProcessing(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Resume Optimizer</h1>
        <p>Upload your resume and job description to get enhancement suggestions</p>
        
        {/* Display API connection status */}
        <div className={`api-status ${apiConnected === true ? 'connected' : apiConnected === false ? 'disconnected' : 'unknown'}`}>
          {apiConnected === true ? 'API Connected ✓' : 
           apiConnected === false ? 'API Disconnected ✗' : 
           'Checking API connection...'}
        </div>
      </header>
      
      <main className="app-main">
        {error && (
          <div className="error-message">
            <h3>Error</h3>
            <p>{error}</p>
          </div>
        )}
        
        <div className="input-section">
          <div className="left-side">
            <FileUpload 
              onFileUpload={handleFileUpload} 
              parsedResume={parsedResume}
            />
          </div>
          
          <div className="right-side">
            <JobDescription 
              onSubmit={handleOptimize}
              isResumeUploaded={!!resumeId}
              isProcessing={isProcessing}
            />
          </div>
        </div>
        
        <ProcessSteps 
          currentStep={currentStep}
          stats={processStats}
        />
        
        {keywords.length > 0 && (
          <KeywordResults keywords={keywords} />
        )}
        
        {modifications.length > 0 && (
          <Results 
            modifications={modifications}
            resumeId={resumeId}
            statistics={statistics}
          />
        )}
        
        <DebugView responses={apiResponses} />
      </main>
      
      <footer className="app-footer">
        <p>&copy; {new Date().getFullYear()} Resume Optimizer - Demo Version</p>
      </footer>
    </div>
  )
}

export default App

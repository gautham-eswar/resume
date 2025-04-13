// API Service for Resume Optimizer

// API base URL (use full URL with correct protocol)
const API_BASE_URL = 'http://127.0.0.1:5000';

/**
 * Test API connectivity
 * @returns {Promise<Object>} Connection status
 */
export async function testApiConnection() {
  try {
    console.log('Testing API connection to:', `${API_BASE_URL}/api/test`);
    const response = await fetch(`${API_BASE_URL}/api/test`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
      mode: 'cors'  // Explicitly set CORS mode
    });
    
    if (!response.ok) {
      throw new Error(`Server responded with status: ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('API connection test failed:', error);
    throw error;
  }
}

/**
 * Upload resume file to the backend
 * @param {File} file - The resume file to upload
 * @returns {Promise<Object>} The parsed resume data
 */
export async function uploadResume(file) {
  console.log('Uploading resume to:', `${API_BASE_URL}/api/upload`);
  
  const formData = new FormData();
  formData.append('resume', file);
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: 'POST',
      body: formData,
      mode: 'cors',  // Explicitly set CORS mode
      credentials: 'omit'  // Don't send credentials
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      console.error('Upload request failed:', response.status, data);
      throw new Error(data.error || `Server responded with status: ${response.status}`);
    }
    
    return data;
  } catch (error) {
    console.error('Resume upload failed:', error);
    throw error;
  }
}

/**
 * Optimize resume with job description
 * @param {string} resumeId - The ID of the uploaded resume
 * @param {string} jobDescription - The job description text
 * @returns {Promise<Object>} The optimization results
 */
export async function optimizeResume(resumeId, jobDescription) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/optimize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ resumeId, jobDescription }),
      mode: 'cors'  // Explicitly set CORS mode
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.error || `Server responded with status: ${response.status}`);
    }
    
    return data;
  } catch (error) {
    console.error('Resume optimization failed:', error);
    throw error;
  }
}

/**
 * Get download URL for enhanced resume
 * @param {string} resumeId - The ID of the resume
 * @param {string} format - The format (json or pdf)
 * @returns {string} The download URL
 */
export function getDownloadUrl(resumeId, format) {
  return `${API_BASE_URL}/api/download/${resumeId}/${format}`;
} 
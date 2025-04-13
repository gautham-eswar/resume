import { useState } from 'react';

function FileUpload({ onFileUpload, parsedResume }) {
  const [file, setFile] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [showParsedResume, setShowParsedResume] = useState(false);
  const [uploadDetails, setUploadDetails] = useState(null);

  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      const selectedFile = e.target.files[0];
      setFile(selectedFile);
      
      // Display file details
      setUploadDetails({
        name: selectedFile.name,
        type: selectedFile.type,
        size: `${Math.round(selectedFile.size / 1024)} KB`,
        lastModified: new Date(selectedFile.lastModified).toLocaleString()
      });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    setIsUploading(true);
    try {
      console.log("Uploading file:", file.name);
      console.log("File size:", file.size, "bytes");
      console.log("File type:", file.type);
      
      // Check if file type is supported
      if (!['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'].includes(file.type)) {
        console.warn("Warning: File type may not be supported by the backend");
      }
      
      // Add a delay to make sure console logs appear
      await new Promise(resolve => setTimeout(resolve, 500));
      
      const response = await onFileUpload(file);
      console.log("Upload successful:", response);
    } catch (error) {
      console.error('Error uploading file:', error);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="file-upload-container">
      <h2>Upload Resume</h2>
      <form onSubmit={handleSubmit}>
        <div className="file-input-wrapper">
          <input 
            type="file" 
            onChange={handleFileChange} 
            accept=".pdf,.docx"
            disabled={isUploading}
          />
          <button 
            type="submit" 
            disabled={!file || isUploading}
            className="upload-btn"
          >
            {isUploading ? 'Uploading...' : 'Upload Resume'}
          </button>
        </div>
      </form>

      {file && (
        <div className="file-info">
          <p>Selected file: <strong>{file.name}</strong></p>
          
          {uploadDetails && (
            <div className="upload-details">
              <p><strong>File Details:</strong></p>
              <ul>
                <li>Type: {uploadDetails.type}</li>
                <li>Size: {uploadDetails.size}</li>
                <li>Last modified: {uploadDetails.lastModified}</li>
              </ul>
            </div>
          )}
        </div>
      )}

      {parsedResume && (
        <div className="parsed-resume">
          <div className="collapsible-header" onClick={() => setShowParsedResume(!showParsedResume)}>
            <h3>Parsed Resume {showParsedResume ? '▼' : '►'}</h3>
          </div>
          {showParsedResume && (
            <div className="collapsible-content">
              <pre>{JSON.stringify(parsedResume, null, 2)}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default FileUpload; 
# Resume Optimizer

A full-stack application that helps users optimize their resumes for specific job descriptions. The system extracts keywords from job descriptions, analyzes resumes, and suggests enhancements to increase the match rate with job requirements.

## Project Structure

This project consists of two main components:

### 1. Python Backend
- `app.py` - Flask API that connects the frontend to the Python backend
- `optimizer.py` - Main entry point for the resume optimization pipeline
- `resume_parser.py` - Extracts and structures text from resume documents
- `keyword_extractor.py` - Extracts relevant keywords from job descriptions
- `embeddings.py` - Handles semantic matching between resumes and keywords
- `enhancer.py` - Enhances resume bullets with targeted keywords
- `PDFsave.py` - Saves enhanced resumes to PDF format

### 2. React Frontend
- `/resume-optimizer/` - React application for the user interface
  - `/src/components/` - UI components
  - `/src/services/` - API service for backend communication

## Prerequisites

- Python 3.8+
- Node.js 16+
- npm 8+
- OpenAI API key

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/resume-optimizer.git
   cd resume-optimizer
   ```

2. Run the setup script to install dependencies:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

3. Set your OpenAI API key:
   - Open `optimizer.py`
   - Replace `"YOUR_OPENAI_API_KEY_HERE"` with your actual OpenAI API key

## Running the Application

1. Start the Flask backend:
   ```
   source venv/bin/activate
   python app.py
   ```

2. In a separate terminal, start the React frontend:
   ```
   cd resume-optimizer
   npm run dev
   ```

3. Open your browser and go to http://localhost:5173

## Using the Application

1. Upload your resume (PDF or DOCX format)
2. Paste the job description
3. Click "Analyze Resume"
4. View the optimization results:
   - Extracted keywords and their relevance
   - Matches found in your resume
   - Enhanced bullet points
5. Download the enhanced resume in JSON or PDF format

## Architecture

```
┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│  React        │     │  Flask API   │     │  OpenAI API  │
│  Frontend     │◄────┤  Backend     │◄────┤              │
│               │     │              │     │              │
└───────┬───────┘     └──────┬───────┘     └──────────────┘
        │                    │
        ▼                    ▼
┌───────────────┐     ┌──────────────┐
│  User         │     │  Resume      │
│  Interface    │     │  Processing  │
│               │     │  Pipeline    │
└───────────────┘     └──────────────┘
```

## API Endpoints

- `POST /api/upload` - Upload and parse resume
- `POST /api/optimize` - Optimize resume with job description
- `GET /api/download/:resumeId/:format` - Download enhanced resume

## Libraries Used

### Backend
- Flask - Web framework
- OpenAI - AI-powered text processing
- NLTK - Natural language processing
- scikit-learn - Machine learning utilities
- pandas - Data processing
- PyPDF & docx2txt - Document parsing

### Frontend
- React - UI library
- Vite - Build tool

## License

MIT License 
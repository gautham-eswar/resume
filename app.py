"""
Flask API for Resume Optimizer

This script provides API endpoints that connect to the Python backend
for resume optimization.
"""

import os
import uuid
import json
import logging
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
from typing import Dict, Any, Tuple, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import the optimizer
from optimizer import ResumeOptimizationPipeline

# Initialize Flask app
app = Flask(__name__)

# Configure CORS for Lovable frontend
FRONTEND_URL = os.getenv('FRONTEND_URL', '*')
CORS(app, resources={
    r"/api/*": {
        "origins": FRONTEND_URL,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "X-Total-Count"]
    }
})

# Configure logging based on environment
if os.getenv('FLASK_ENV') == 'production':
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
else:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

# Configure uploads with environment-specific paths
if os.getenv('FLASK_ENV') == 'production':
    # In production, use /tmp directory for uploads
    UPLOAD_FOLDER = '/tmp/uploads'
    OUTPUT_FOLDER = '/tmp/output'
else:
    # In development, use local directories
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Dictionary to store resume data by ID with TTL cleanup
from datetime import datetime, timedelta

class ResumeStore:
    def __init__(self):
        self.store: Dict[str, Dict[str, Any]] = {}
        self.ttl_hours = 24  # Store resumes for 24 hours

    def add(self, resume_id: str, data: Dict[str, Any]):
        self.store[resume_id] = {
            **data,
            'timestamp': datetime.now()
        }
        self._cleanup()

    def get(self, resume_id: str) -> Dict[str, Any]:
        if resume_id in self.store:
            data = self.store[resume_id]
            if datetime.now() - data['timestamp'] < timedelta(hours=self.ttl_hours):
                return data
            else:
                del self.store[resume_id]
        return None

    def _cleanup(self):
        """Remove entries older than TTL"""
        current_time = datetime.now()
        expired_ids = [
            rid for rid, data in self.store.items()
            if current_time - data['timestamp'] >= timedelta(hours=self.ttl_hours)
        ]
        for rid in expired_ids:
            del self.store[rid]
            # Cleanup files
            try:
                file_path = os.path.join(UPLOAD_FOLDER, f"{rid}_*")
                output_dir = os.path.join(OUTPUT_FOLDER, rid)
                os.system(f"rm {file_path}")
                os.system(f"rm -rf {output_dir}")
            except Exception as e:
                logger.warning(f"Error cleaning up files for {rid}: {str(e)}")

resume_store = ResumeStore()

def create_response(data: Any = None, error: str = None, status: int = 200) -> Tuple[Dict[str, Any], int]:
    """Standardize API responses"""
    response = {
        "success": error is None,
        "data": data if data is not None else {},
        "error": error if error is not None else None
    }
    return jsonify(response), status

def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(Exception)
def handle_error(error: Exception) -> Tuple[Dict[str, Any], int]:
    """Global error handler"""
    logger.error(f"Unexpected error: {str(error)}", exc_info=True)
    return create_response(error=f"Internal server error: {str(error)}", status=500)

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        "status": "ok",
        "message": "Resume Optimizer API is running"
    })

@app.route('/api/health', methods=['GET'])
def health_check() -> Tuple[Dict[str, Any], int]:
    """Health check endpoint"""
    return create_response(data={"status": "healthy", "version": "1.0.0"})

@app.route('/api/upload', methods=['POST'])
def upload_resume() -> Tuple[Dict[str, Any], int]:
    """Upload and parse a resume"""
    try:
        if 'resume' not in request.files:
            return create_response(error="No resume file provided", status=400)

        file = request.files['resume']
        if file.filename == '':
            return create_response(error="No file selected", status=400)

        if not allowed_file(file.filename):
            return create_response(
                error="File type not supported. Please upload PDF, DOCX, or TXT.",
                status=400
            )

        # Generate unique ID and save file
        resume_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{resume_id}_{filename}")
        file.save(file_path)

        # Initialize pipeline and parse resume
        pipeline = ResumeOptimizationPipeline(verbose=True)
        resume_text = pipeline.parser.extract_text_from_file(file_path)
        resume_data = pipeline.parser.parse_resume(resume_text)

        # Store resume data
        resume_store.add(resume_id, {
            'file_path': file_path,
            'data': resume_data
        })

        return create_response(data={
            'resumeId': resume_id,
            'parsedResume': resume_data
        })

    except Exception as e:
        logger.error(f"Error during resume upload: {str(e)}", exc_info=True)
        return create_response(error=f"Error parsing resume: {str(e)}", status=500)

@app.route('/api/optimize', methods=['POST'])
def optimize_resume() -> Tuple[Dict[str, Any], int]:
    """Optimize resume with job description"""
    try:
        data = request.get_json()
        
        # Validate inputs
        if not data:
            return create_response(error="No data provided", status=400)
            
        resume_id = data.get('resumeId')
        job_description = data.get('jobDescription')
        
        if not resume_id:
            return create_response(error="Resume ID is required", status=400)
        if not job_description:
            return create_response(error="Job description is required", status=400)
        if not resume_store.get(resume_id):
            return create_response(error="Resume not found. Please upload again.", status=404)

        # Get resume data and setup
        resume_data = resume_store.get(resume_id)['data']
        resume_file_path = resume_store.get(resume_id)['file_path']
        output_dir = os.path.join(OUTPUT_FOLDER, resume_id)
        os.makedirs(output_dir, exist_ok=True)

        # Run optimization
        pipeline = ResumeOptimizationPipeline(verbose=True)
        optimization_result = pipeline.run_pipeline(
            resume_file_path=resume_file_path,
            job_description_text=job_description,
            output_dir=output_dir
        )

        return create_response(data={
            'optimizationResult': optimization_result,
            'outputDirectory': output_dir
        })

    except Exception as e:
        logger.error(f"Error during optimization: {str(e)}", exc_info=True)
        return create_response(error=f"Error optimizing resume: {str(e)}", status=500)

@app.route('/api/download/<resume_id>/<format>', methods=['GET'])
def download_resume(resume_id: str, format: str) -> Union[Tuple[Dict[str, Any], int], Any]:
    """Download enhanced resume in specified format"""
    try:
        if not resume_store.get(resume_id):
            return create_response(error="Resume not found", status=404)

        output_dir = os.path.join(OUTPUT_FOLDER, resume_id)
        if format.lower() == 'json':
            file_path = os.path.join(output_dir, 'enhanced_resume.json')
        elif format.lower() == 'pdf':
            file_path = os.path.join(output_dir, 'enhanced_resume.pdf')
        else:
            return create_response(error="Invalid format. Use 'json' or 'pdf'.", status=400)

        if not os.path.exists(file_path):
            return create_response(error=f"Enhanced resume in {format} format not found", status=404)

        return send_file(
            file_path,
            as_attachment=True,
            download_name=f"enhanced_resume.{format.lower()}"
        )

    except Exception as e:
        logger.error(f"Error downloading resume: {str(e)}", exc_info=True)
        return create_response(error=f"Error downloading resume: {str(e)}", status=500)

# Only run the app if this file is being run directly
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=port)
    else:
        app.run(debug=True, host='0.0.0.0', port=port) 
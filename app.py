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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("app")

# Load environment variables
load_dotenv()

# Import the optimizer
from optimizer import ResumeOptimizationPipeline

# Initialize Flask app
app = Flask(__name__)

# Add detailed request logging
@app.before_request
def log_request_info():
    logger.info(f"Request: {request.method} {request.path}")
    logger.info(f"Headers: {dict(request.headers)}")
    if request.is_json:
        logger.info(f"JSON Body: {request.json}")

# Configure CORS for Lovable frontend
FRONTEND_URL = os.getenv('FRONTEND_URL', '*')
CORS(app, resources={
    r"/api/*": {
        "origins": FRONTEND_URL,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "X-Total-Count"]
    },
    r"/optimize": {
        "origins": FRONTEND_URL,
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Type", "X-Total-Count"]
    }
})

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
        "message": "Resume Optimizer API is running",
        "endpoints": [
            {"path": "/api/optimize", "method": "POST", "description": "Optimize resume with job description (API path)"},
            {"path": "/optimize", "method": "POST", "description": "Optimize resume with job description (alternate path)"},
            {"path": "/api/upload", "method": "POST", "description": "Upload and parse a resume"},
            {"path": "/api/download/<resume_id>/<format>", "method": "GET", "description": "Download enhanced resume"}
        ]
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
        # Parse request data
        data = request.get_json()
        if not data or 'jobDescription' not in data:
            return jsonify({'error': 'Job description is required'}), 400
        
        job_description = data['jobDescription']
        
        # Log received data
        logger.info(f"Received optimization request with job description of length: {len(job_description)}")
        
        # Handle resume URL if provided
        resume_file_path = None
        if 'resumeUrl' in data and data['resumeUrl']:
            try:
                # Download file from URL
                resume_url = data['resumeUrl']
                logger.info(f"Resume URL provided: {resume_url}")
                
                # Download the file
                import requests
                response = requests.get(resume_url)
                if response.status_code == 200:
                    # Save to temp file
                    import tempfile
                    
                    # Determine file extension
                    file_extension = '.pdf'  # Default
                    if '.docx' in resume_url.lower():
                        file_extension = '.docx'
                    elif '.txt' in resume_url.lower():
                        file_extension = '.txt'
                    
                    # Create temp dir if not exists
                    temp_dir = os.path.join(tempfile.gettempdir(), 'resume_optimizer')
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # Save file
                    resume_file_path = os.path.join(temp_dir, f"{uuid.uuid4()}{file_extension}")
                    with open(resume_file_path, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"Resume downloaded and saved to: {resume_file_path}")
                else:
                    logger.error(f"Failed to download resume: Status code {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading resume: {str(e)}")
        # Check for resumeId if URL wasn't provided
        elif 'resumeId' in data and data['resumeId']:
            resume_id = data['resumeId']
            if not resume_store.get(resume_id):
                return create_response(error="Resume not found. Please upload again.", status=404)
            resume_file_path = resume_store.get(resume_id)['file_path']
            logger.info(f"Using previously uploaded resume with ID: {resume_id}")
        
        # Create output directory
        import tempfile
        import uuid
        output_id = str(uuid.uuid4())
        if 'resumeId' in data and data['resumeId'] and resume_store.get(data['resumeId']):
            output_dir = os.path.join(OUTPUT_FOLDER, data['resumeId'])
        else:
            output_dir = os.path.join(tempfile.gettempdir(), 'resume_output', output_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize the optimization pipeline
        pipeline = ResumeOptimizationPipeline(verbose=True)
        
        # Run the pipeline
        if resume_file_path:
            # Full optimization with resume
            logger.info("Running full optimization pipeline with resume")
            
            summary = pipeline.run_pipeline(
                resume_file_path=resume_file_path,
                job_description_text=job_description,
                output_dir=output_dir
            )
            
            # Format response
            response_data = {
                'success': True,
                'message': 'Resume optimization complete',
                'optimizedContent': format_optimization_results(pipeline),
                'statistics': summary['statistics'],
                'analysis': {
                    'keywords': pipeline.keywords_data['keywords'],
                    'semantic_matches': pipeline.semantic_matches['similarity_results'],
                    'modifications': pipeline.modifications
                }
            }
        else:
            # Keywords only (no resume)
            logger.info("Running keyword extraction only (no resume provided)")
            
            # Extract keywords from job description
            pipeline.keywords_data = pipeline.keyword_extractor.extract_keywords(job_description)
            
            # Format response for keywords only
            response_data = {
                'success': True,
                'message': 'Job description analyzed (no resume provided)',
                'optimizedContent': format_keywords_only_results(pipeline.keywords_data),
                'statistics': {
                    'keywords_extracted': len(pipeline.keywords_data['keywords']),
                },
                'analysis': {
                    'keywords': pipeline.keywords_data['keywords']
                }
            }
        
        # Clean up if we downloaded from URL
        if 'resumeUrl' in data and data['resumeUrl'] and resume_file_path and os.path.exists(resume_file_path):
            try:
                os.remove(resume_file_path)
                logger.info(f"Deleted temporary file: {resume_file_path}")
            except:
                logger.warning(f"Failed to delete temporary file: {resume_file_path}")
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        logger.error(f"Error in optimization process: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': f'Error in optimization process: {str(e)}',
            'error_details': traceback.format_exc()
        }), 500

# Helper function to format optimization results as markdown
def format_optimization_results(pipeline):
    # Group keywords by relevance
    keywords_by_relevance = {'high': [], 'medium': [], 'low': []}
    for kw in pipeline.keywords_data['keywords']:
        relevance = kw['relevance_score']
        if relevance >= 8:
            keywords_by_relevance['high'].append(kw)
        elif relevance >= 5:
            keywords_by_relevance['medium'].append(kw)
        else:
            keywords_by_relevance['low'].append(kw)
    
    # Create high relevance keywords section
    high_relevance_section = ""
    for kw in keywords_by_relevance['high']:
        high_relevance_section += f"- **{kw['keyword']}** - {kw['context']}\n"
    
    # Create medium relevance keywords section
    medium_relevance_section = ""
    for kw in keywords_by_relevance['medium']:
        medium_relevance_section += f"- **{kw['keyword']}** - {kw['context']}\n"
    
    # Create keyword gap analysis table
    gap_analysis_table = ""
    for match in pipeline.semantic_matches.get('similarity_results', [])[:10]:
        presence = "Yes" if match['similarity_score'] > 0.75 else "No"
        gap_analysis_table += f"| **{match['keyword']}** | {presence} | {match['similarity_score']:.2f} |\n"
    
    # Create bullet point improvements section
    improvements_section = ""
    for mod in pipeline.modifications[:5]:
        improvements_section += f"### Original:\n{mod['original_bullet']}\n\n### Enhanced:\n{mod['enhanced_bullet']}\n\n"
    
    # Create final report
    report = f"""
## KEY SKILLS ANALYSIS

### Must-Have Skills (High Relevance)
{high_relevance_section}

### Nice-to-Have Skills (Medium Relevance)
{medium_relevance_section}

## KEYWORD GAP ANALYSIS

| Skill | Present in Resume | Similarity Score |
|-------|-------------------|-----------------|
{gap_analysis_table}

## RECOMMENDED BULLET POINT IMPROVEMENTS

{improvements_section}

## GENERAL RECOMMENDATIONS

1. Tailor your skills section to highlight the must-have skills identified above
2. Quantify your achievements with specific metrics where possible
3. Use industry-specific terminology that matches the job description
4. Focus on recent and relevant experiences that align with the job requirements
5. Ensure your resume passes ATS scanning by using exact keyword matches
"""
    return report

# Helper function to format keywords-only results
def format_keywords_only_results(keywords_data):
    # Group keywords by relevance
    keywords_by_relevance = {'high': [], 'medium': [], 'low': []}
    for kw in keywords_data['keywords']:
        relevance = kw['relevance_score']
        if relevance >= 8:
            keywords_by_relevance['high'].append(kw)
        elif relevance >= 5:
            keywords_by_relevance['medium'].append(kw)
        else:
            keywords_by_relevance['low'].append(kw)
    
    # Create high relevance keywords section
    high_relevance_section = ""
    for kw in keywords_by_relevance['high']:
        high_relevance_section += f"- **{kw['keyword']}** - {kw['context']}\n"
    
    # Create medium relevance keywords section
    medium_relevance_section = ""
    for kw in keywords_by_relevance['medium']:
        medium_relevance_section += f"- **{kw['keyword']}** - {kw['context']}\n"
    
    # Create final report
    report = f"""
## KEY SKILLS ANALYSIS

### Must-Have Skills (High Relevance)
{high_relevance_section}

### Nice-to-Have Skills (Medium Relevance)
{medium_relevance_section}

## RECOMMENDED IMPROVEMENTS

1. Ensure your resume includes the must-have skills listed above
2. Use exact keyword matches for technical skills to pass ATS screening
3. Quantify achievements related to these skills with specific metrics
4. Highlight experiences that demonstrate proficiency in these areas
5. Consider creating a skills section that prominently features these keywords
"""
    return report

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

@app.route('/optimize', methods=['POST', 'OPTIONS'])
def optimize_resume_alt():
    """Alternative endpoint for optimize - mirrors /api/optimize"""
    if request.method == 'OPTIONS':
        # Handle OPTIONS request (CORS preflight)
        response = jsonify({'success': True})
        return response
    return optimize_resume()

# ASGI wrapper for deployment
import asgiref.wsgi
app = asgiref.wsgi.WsgiToAsgi(app.wsgi_app)

# Only run the app if this file is being run directly
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    if os.getenv('FLASK_ENV') == 'production':
        app.run(host='0.0.0.0', port=port)
    else:
        app.run(debug=True, host='0.0.0.0', port=port) 
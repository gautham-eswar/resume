"""
API server for Resume Optimizer.

This script provides a simple Flask API that exposes the resume optimization
functionality to the web frontend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from resume_enhancer import enhance_resume
import logging
import json
import tempfile
import uuid
from werkzeug.utils import secure_filename

# Import your existing pipeline
from optimizer import ResumeOptimizationPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure OpenAI API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable must be set")

# Create temp directories
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), 'resume_uploads')
OUTPUT_DIR = os.path.join(tempfile.gettempdir(), 'resume_outputs')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

# Ensure upload and output directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/api/optimize', methods=['POST'])
def optimize_resume():
    try:
        data = request.get_json()
        resume_data = data.get('resume')
        job_description = data.get('jobDescription')
        
        if not resume_data or not job_description:
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Generate unique ID for this optimization
        optimization_id = os.urandom(16).hex()
        output_dir = os.path.join(OUTPUT_FOLDER, optimization_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Enhance resume
        enhanced_resume, modifications = enhance_resume(resume_data, job_description)
        
        # Save results
        with open(os.path.join(output_dir, 'enhanced_resume.json'), 'w') as f:
            json.dump(enhanced_resume, f, indent=2)
            
        with open(os.path.join(output_dir, 'modifications.json'), 'w') as f:
            json.dump(modifications, f, indent=2)
        
        return jsonify({
            'success': True,
            'enhanced_resume': enhanced_resume,
            'modifications': modifications,
            'optimization_id': optimization_id
        })
        
    except Exception as e:
        logging.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500

def format_optimization_results(pipeline):
    """Format optimization results as markdown for the frontend"""
    # Keywords section
    keywords_by_relevance = {'high': [], 'medium': [], 'low': []}
    for kw in pipeline.keywords_data['keywords']:
        relevance = kw['relevance_score']
        if relevance >= 8:
            keywords_by_relevance['high'].append(kw)
        elif relevance >= 5:
            keywords_by_relevance['medium'].append(kw)
        else:
            keywords_by_relevance['low'].append(kw)
    
    # Create markdown report
    report = f"""
## KEY SKILLS ANALYSIS

### Must-Have Skills (High Relevance)
{"".join([f"- **{kw['keyword']}** - {kw['context']}\n" for kw in keywords_by_relevance['high']])}

### Nice-to-Have Skills (Medium Relevance)
{"".join([f"- **{kw['keyword']}** - {kw['context']}\n" for kw in keywords_by_relevance['medium']])}

## KEYWORD GAP ANALYSIS

| Skill | Present in Resume | Similarity Score |
|-------|-------------------|-----------------|
{"".join([f"| **{match['keyword']}** | {'Yes' if match['similarity_score'] > 0.75 else 'No'} | {match['similarity_score']:.2f} |\n" for match in pipeline.semantic_matches.get('similarity_results', [])[:10]])}

## RECOMMENDED BULLET POINT IMPROVEMENTS

{"".join([f"### Original:\n{mod['original_bullet']}\n\n### Enhanced:\n{mod['enhanced_bullet']}\n\n" for mod in pipeline.modifications[:5]])}

## GENERAL RECOMMENDATIONS

1. Tailor your skills section to highlight the must-have skills identified above
2. Quantify your achievements with specific metrics where possible
3. Use industry-specific terminology that matches the job description
4. Focus on recent and relevant experiences that align with the job requirements
5. Ensure your resume passes ATS scanning by using exact keyword matches
"""
    return report

def format_job_description_results(keywords_data):
    """Format job description analysis as markdown for the frontend"""
    # Keywords section
    keywords_by_relevance = {'high': [], 'medium': [], 'low': []}
    for kw in keywords_data['keywords']:
        relevance = kw['relevance_score']
        if relevance >= 8:
            keywords_by_relevance['high'].append(kw)
        elif relevance >= 5:
            keywords_by_relevance['medium'].append(kw)
        else:
            keywords_by_relevance['low'].append(kw)
    
    # Create markdown report
    report = f"""
## KEY SKILLS ANALYSIS

### Must-Have Skills (High Relevance)
{"".join([f"- **{kw['keyword']}** - {kw['context']}\n" for kw in keywords_by_relevance['high']])}

### Nice-to-Have Skills (Medium Relevance)
{"".join([f"- **{kw['keyword']}** - {kw['context']}\n" for kw in keywords_by_relevance['medium']])}

## RECOMMENDED IMPROVEMENTS

1. Ensure your resume includes the must-have skills listed above
2. Use exact keyword matches for technical skills to pass ATS screening
3. Quantify achievements related to these skills with specific metrics
4. Highlight experiences that demonstrate proficiency in these areas
5. Consider creating a skills section that prominently features these keywords
"""
    return report

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=True) 
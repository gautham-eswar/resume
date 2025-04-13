"""
API server for Resume Optimizer.

This script provides a simple Flask API that exposes the resume optimization
functionality to the web frontend.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import tempfile
import uuid
from werkzeug.utils import secure_filename

# Import your existing pipeline
from optimizer import ResumeOptimizationPipeline

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

@app.route('/optimize', methods=['POST'])
def optimize_resume():
    # Get job description from request
    data = request.json
    if not data or 'jobDescription' not in data:
        return jsonify({'error': 'Job description is required'}), 400
    
    job_description = data['jobDescription']
    
    # Handle resume URL if provided
    resume_file_path = None
    if 'resumeUrl' in data and data['resumeUrl']:
        try:
            # Download file from URL
            import requests
            resume_url = data['resumeUrl']
            response = requests.get(resume_url)
            if response.status_code == 200:
                # Save to temp file
                file_extension = '.pdf'  # Default to PDF
                if '.docx' in resume_url.lower():
                    file_extension = '.docx'
                elif '.txt' in resume_url.lower():
                    file_extension = '.txt'
                
                # Create unique filename
                resume_file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{file_extension}")
                with open(resume_file_path, 'wb') as f:
                    f.write(response.content)
        except Exception as e:
            return jsonify({'error': f'Error downloading resume: {str(e)}'}), 500
    
    try:
        # Create unique output directory
        result_id = str(uuid.uuid4())
        output_dir = os.path.join(OUTPUT_DIR, result_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize pipeline
        pipeline = ResumeOptimizationPipeline(api_key=OPENAI_API_KEY, verbose=True)
        
        if resume_file_path:
            # Run full pipeline with resume
            result = pipeline.run_pipeline(
                resume_file_path=resume_file_path,
                job_description_text=job_description,
                output_dir=output_dir
            )
            
            # Format results for frontend
            response_data = {
                'success': True,
                'message': 'Resume optimized successfully',
                'optimizedContent': format_optimization_results(pipeline),
                'statistics': result['statistics'],
                'analysis': {
                    'keywords': pipeline.keywords_data['keywords'],
                    'matches': pipeline.semantic_matches['similarity_results'],
                    'modifications': pipeline.modifications
                }
            }
        else:
            # Extract keywords without resume
            pipeline.keywords_data = pipeline.keyword_extractor.extract_keywords(job_description)
            
            # Format results without resume
            response_data = {
                'success': True,
                'message': 'Job description analyzed successfully',
                'optimizedContent': format_job_description_results(pipeline.keywords_data),
                'statistics': {
                    'keywords_extracted': len(pipeline.keywords_data['keywords']),
                },
                'analysis': {
                    'keywords': pipeline.keywords_data['keywords']
                }
            }
        
        return jsonify(response_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error in optimization process: {str(e)}'}), 500
    finally:
        # Clean up temp files
        if resume_file_path and os.path.exists(resume_file_path):
            try:
                os.remove(resume_file_path)
            except:
                pass

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
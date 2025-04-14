import os
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def extract_keywords(job_description):
    """Extract important keywords from job description."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume writer specializing in keyword extraction."},
                {"role": "user", "content": f"""
                Extract the most important keywords from this job description that should be included in a resume.
                Focus on:
                1. Technical skills
                2. Soft skills
                3. Industry-specific terms
                4. Leadership and management terms
                5. Tools and technologies
                
                Job Description:
                {job_description}
                
                Return the keywords as a JSON list.
                """}
            ],
            temperature=0.3
        )
        
        keywords = json.loads(response.choices[0].message.content)
        logger.info(f"Extracted {len(keywords)} keywords from job description")
        return keywords
        
    except Exception as e:
        logger.error(f"Error extracting keywords: {str(e)}")
        raise

def enhance_bullet_point(bullet, keywords):
    """Enhance a single bullet point with keywords."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional resume writer specializing in keyword optimization while maintaining factual accuracy."},
                {"role": "user", "content": f"""
                Task: Enhance the following resume bullet point by naturally incorporating the specified keywords.

                Original bullet point:
                "{bullet}"

                Keywords to incorporate naturally:
                {json.dumps(keywords, indent=2)}

                Requirements:
                1. MUST include ALL the keywords in the enhanced bullet point
                2. MUST preserve ALL numbers, percentages, and metrics EXACTLY as they appear
                3. MUST maintain the original meaning, achievements, and scope of work
                4. MUST keep the same professional tone and tense
                5. Changes should be minimal and natural - only make changes needed to incorporate keywords
                6. Final bullet MUST sound natural and professional
                7. If impossible to include all keywords naturally, prioritize the ones listed first

                Enhanced bullet point:
                """}
            ],
            max_tokens=512,
            temperature=0.3
        )
        
        enhanced_bullet = response.choices[0].message.content.strip().strip('"')
        return enhanced_bullet
        
    except Exception as e:
        logger.error(f"Error enhancing bullet point: {str(e)}")
        raise

def enhance_resume(resume_data, job_description):
    """Enhance resume with keywords from job description."""
    try:
        # Extract keywords from job description
        keywords = extract_keywords(job_description)
        
        # Track modifications
        modifications = {
            'original_bullets': [],
            'enhanced_bullets': [],
            'keywords_added': []
        }
        
        # Enhance each bullet point
        enhanced_resume = resume_data.copy()
        for section in enhanced_resume.get('sections', []):
            if 'bullets' in section:
                enhanced_bullets = []
                for bullet in section['bullets']:
                    original_bullet = bullet
                    enhanced_bullet = enhance_bullet_point(bullet, keywords)
                    
                    if enhanced_bullet != original_bullet:
                        modifications['original_bullets'].append(original_bullet)
                        modifications['enhanced_bullets'].append(enhanced_bullet)
                        modifications['keywords_added'].extend(
                            [k for k in keywords if k.lower() in enhanced_bullet.lower() and k.lower() not in original_bullet.lower()]
                        )
                    
                    enhanced_bullets.append(enhanced_bullet)
                section['bullets'] = enhanced_bullets
        
        logger.info(f"Resume enhancement complete. Modified {len(modifications['enhanced_bullets'])} bullets.")
        return enhanced_resume, modifications
        
    except Exception as e:
        logger.error(f"Error enhancing resume: {str(e)}")
        raise

if __name__ == '__main__':
    # Test the enhancement
    test_resume = {
        "sections": [{
            "title": "Experience",
            "bullets": [
                "Led a team of 5 developers to deliver a web application",
                "Improved system performance by 40% through optimization"
            ]
        }]
    }
    
    test_job = """
    We are looking for a Senior Software Engineer with:
    - Strong Python experience
    - Team leadership skills
    - Performance optimization expertise
    """
    
    enhanced, mods = enhance_resume(test_resume, test_job)
    print(json.dumps(enhanced, indent=2))
    print("\nModifications:", json.dumps(mods, indent=2)) 
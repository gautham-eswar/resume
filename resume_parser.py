"""
Resume parser module.

This module extracts text from resume files (PDF, DOCX, TXT)
and parses it into structured JSON format.
"""

import os
import json
import re
import logging
from typing import Dict, List, Any, Optional

# Import document libraries (if available)
try:
    import PyPDF2
except ImportError:
    pass

try:
    import docx
except ImportError:
    pass

# Import OpenAI
try:
    from openai import OpenAI
except ImportError:
    raise ImportError("OpenAI Python package is required. Install with: pip install openai")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("resume_parser")


class ResumeParser:
    """
    Parse resume files into structured JSON format.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the ResumeParser with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as a parameter or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)
    
    def extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various resume file formats (PDF, DOCX, TXT).
        
        Args:
            file_path: Path to the resume file
            
        Returns:
            str: Extracted text from the resume
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Resume file not found: {file_path}")
            
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self._extract_from_docx(file_path)
            elif file_extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
        except Exception as e:
            logger.error(f"Error extracting text from resume: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        except NameError:
            raise ImportError("PyPDF2 is not installed. Please install with: pip install PyPDF2")
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file."""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except NameError:
            raise ImportError("python-docx is not installed. Please install with: pip install python-docx")
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def parse_resume(self, resume_text: str) -> Dict[str, Any]:
        """
        Parse resume text into structured JSON using OpenAI API.
        
        Args:
            resume_text: Text extracted from resume
            
        Returns:
            dict: Structured resume data
        """
        logger.info("Parsing resume text into structured data")
        
        # Define the structure we want from the model
        prompt = f"""
        Parse the following resume text into a structured JSON format. Include the following sections:
        1. Personal Information (name, email, phone, location, website/LinkedIn)
        2. Summary/Objective
        3. Skills - categorize skills into:
           - Technical Skills: Programming languages, tools, software, technical methodologies
           - Soft Skills: Communication, leadership, teamwork, etc.
        4. Experience - For each position, extract:
           - company
           - title
           - location (city, state, country, and if remote work is mentioned)
           - employment_type (full-time, part-time, contract, internship)
           - dates (start_date, end_date or "Present") (If there's only one date, it's the end_date)
           - responsibilities/achievements (as an array of bullet points)
        5. Education - For each entry, extract:
           - university (institution name)
           - location (city, state, country)
           - degree (type of degree: BA, BS, MS, PhD, etc.)
           - specialization (major/field of study)
           - honors (any honors, distinctions, awards)
           - start_date (year)
           - end_date (year or "Present")
           - gpa (if available)
           - additional_info (courses, activities, or any other relevant information)
        6. Projects (title, description, technologies used) (if the description has multiple bullet points, make sure to include them all in a structured manner)
        7. Certifications/Awards
        8. Languages
        9. Publications - For each publication:
           - title
           - authors
           - journal/conference
           - date
           - url (if available)
        10. Volunteer Experience - For each position:
            - organization
            - role
            - location
            - dates
            - description
        11. Misc (other sections that don't fit above)

        For the Skills section, be very careful to correctly categorize technical vs soft skills.
        Technical skills include specific tools, technologies, programming languages, and technical methodologies.
        Soft skills include interpersonal abilities, communication skills, character traits, and other leadership skills.

        For any missing sections, include them with empty values.
        
        If there's formatting in the resume text, make sure to include it in the JSON output (eg bullet points, sections, etc.)

        Resume text:
        {resume_text}
        
        Provide ONLY the JSON output without any explanation or other text.
        """
        
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional resume parser. Extract structured information from resumes accurately."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=3000,
                response_format={"type": "json_object"}  # Request JSON format
            )
            
            # Extract the response content
            parsed_json_str = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                parsed_json = json.loads(parsed_json_str)
                logger.info("Successfully parsed resume into structured JSON")
                return parsed_json
            except json.JSONDecodeError:
                # Try to extract JSON if surrounded by markdown code blocks
                json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', parsed_json_str)
                if json_match:
                    parsed_json_str = json_match.group(1)
                    return json.loads(parsed_json_str)
                else:
                    logger.error("Failed to parse JSON response from OpenAI")
                    raise Exception("Failed to parse resume into structured format")
                    
        except Exception as e:
            logger.error(f"Error parsing resume: {str(e)}")
            return {
                "error": str(e),
                "status": "failed"
            }
    
    def save_parsed_resume(self, resume_data: Dict[str, Any], output_path: str) -> None:
        """
        Save parsed resume data to JSON file.
        
        Args:
            resume_data: Parsed resume data
            output_path: Path to save parsed resume to
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(resume_data, f, indent=2)
            
        logger.info(f"Parsed resume saved to {output_path}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Parse resume files")
    parser.add_argument("--file", type=str, required=True, help="Path to resume file")
    parser.add_argument("--output", type=str, default="parsed_resume.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Initialize parser
    resume_parser = ResumeParser()
    
    # Extract text
    resume_text = resume_parser.extract_text_from_file(args.file)
    
    # Parse resume
    resume_data = resume_parser.parse_resume(resume_text)
    
    # Save parsed resume
    resume_parser.save_parsed_resume(resume_data, args.output)
    
    print(f"Resume parsed and saved to {args.output}")
"""
Enhanced keyword extraction module for resume optimization.

This module extracts relevant keywords from job descriptions with context and relevance scoring,
preparing them for semantic matching with resume content.
"""

import os
import json
import re
import logging
from typing import Dict, List, Any, Optional, Union

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
logger = logging.getLogger("keyword_extractor")


class KeywordExtractor:
    """Extract and process keywords from job descriptions."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the KeywordExtractor with OpenAI API key.
        
        Args:
            api_key: OpenAI API key. If None, will try to get from environment variable.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Provide it as a parameter or set OPENAI_API_KEY environment variable.")
        
        # Initialize OpenAI client
        try:
            # Try the newer approach
            self.client = OpenAI(api_key=self.api_key)
        except TypeError as e:
            if 'proxies' in str(e):
                # Fallback for older OpenAI versions
                import openai
                openai.api_key = self.api_key
                self.client = openai
            else:
                raise
        
    def extract_keywords(self, job_description: str) -> Dict[str, Any]:
        """
        Extract keywords from job description with relevance scoring and context.
        
        Args:
            job_description: Job description text
            
        Returns:
            dict: Dictionary with keywords and metadata
        """
        logger.info("Extracting keywords from job description")
        
        # Extract raw keywords with context and relevance
        raw_keywords = self._extract_raw_keywords(job_description)
        logger.info(f"Extracted {len(raw_keywords)} raw keywords")
        
        # Filter for quality keywords
        quality_keywords = self._filter_quality_keywords(raw_keywords)
        logger.info(f"Filtered to {len(quality_keywords)} quality keywords")
        
        # Calculate statistics
        stats = self._calculate_statistics(quality_keywords)
        
        return {
            "keywords": quality_keywords,
            "statistics": stats
        }
    
    def _extract_raw_keywords(self, job_description: str) -> List[Dict[str, Any]]:
        """
        Extract raw keywords with context from job description.
        
        Args:
            job_description: Job description text
            
        Returns:
            list: Raw keywords with context and metadata
        """
        # Create the extraction prompt
        prompt = f"""
        Extract relevant keywords from this job description. Follow these instructions EXACTLY:

        1. Extract all professional terms from these categories:
           - Technical skills: tools, languages, platforms, frameworks
           - Domain knowledge: industry-specific concepts 
           - Methodologies/processes: approaches to work
           - Job functions: specific activities or responsibilities
           - Qualifications: degrees, certifications, experience types

        2. For each keyword:
           - Provide the EXACT sentence or context from the job description
           - Assign a relevance score (1-10) based on importance:
             * 10: Explicitly marked as REQUIRED or ESSENTIAL
             * 8-9: Clearly important but not explicitly required
             * 6-7: Clearly mentioned but not emphasized
             * 4-5: Mentioned briefly or as nice-to-have
             * 1-3: Marginally relevant or optional
           - Classify as "hard skill" or "soft skill":
             * Hard skill: Technical abilities, tools, specific knowledge
             * Soft skill: Interpersonal abilities, character traits

        3. Extract specific technologies mentioned in parentheses or after phrases like "including", "such as":
           - Example: "cloud environments (AWS/Azure/GCP)" → Extract "cloud environments", "AWS", "Azure", "GCP"
           - Example: "tools such as JIRA, Confluence" → Extract "JIRA", "Confluence"

        4. For multi-word phrases, extract the most specific form:
           - Example: "machine learning algorithms" is better than just "algorithms"

        5. Ensure every extracted keyword has direct evidence in the text.

        Output as a JSON array with this structure:
        [
          {{
            "keyword": "extracted keyword",
            "context": "exact sentence from job description",
            "relevance_score": number (1-10),
            "skill_type": "hard skill" or "soft skill"
          }}
        ]

        Job Description:
        {job_description}
        """
        
        try:
            # Make the API call with specific JSON formatting instructions
            try:
                # New API format
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a precise keyword extractor for job descriptions. Extract keywords with context and output valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    response_format={"type": "json_object"}  # Request JSON format
                )
                
                # Extract the response content
                content = response.choices[0].message.content.strip()
            except AttributeError:
                # Old API format
                response = self.client.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {
                            "role": "system", 
                            "content": "You are a precise keyword extractor for job descriptions. Extract keywords with context and output valid JSON only."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Low temperature for consistent results
                    response_format={"type": "json_object"}  # Request JSON format
                )
                
                # Extract the response content
                content = response["choices"][0]["message"]["content"].strip()
            
            # Parse the JSON response
            try:
                extracted_data = json.loads(content)
                
                # Handle different possible response structures
                if isinstance(extracted_data, list):
                    keywords = extracted_data
                elif isinstance(extracted_data, dict) and "keywords" in extracted_data:
                    keywords = extracted_data["keywords"]
                else:
                    # Try to find a list in the response
                    for key, value in extracted_data.items():
                        if isinstance(value, list) and len(value) > 0:
                            keywords = value
                            break
                    else:
                        logger.warning("Unexpected JSON structure, couldn't find keywords list")
                        keywords = []
                        
                return keywords
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Try to extract JSON using regex as fallback
                json_match = re.search(r'\[\s*\{.*\}\s*\]', content, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse extracted JSON")
                        
                return self._extract_fallback(job_description)
                
        except Exception as e:
            logger.error(f"Error during keyword extraction: {str(e)}")
            return self._extract_fallback(job_description)
    
    def _extract_fallback(self, job_description: str) -> List[Dict[str, Any]]:
        """
        Fallback method for keyword extraction when API call fails.
        
        Args:
            job_description: Job description text
            
        Returns:
            list: Basic extracted keywords
        """
        logger.info("Using fallback keyword extraction")
        
        # Common skill keywords to look for
        skill_terms = [
            "python", "java", "javascript", "react", "angular", "vue", "node", 
            "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "git",
            "agile", "scrum", "kanban", "jira", "sql", "nosql", "mongodb",
            "data analysis", "machine learning", "ai", "nlp", "computer vision",
            "product management", "ux", "ui", "design thinking", "figma",
            "leadership", "communication", "presentation", "stakeholder management",
            "bachelor", "master", "phd", "mba", "degree"
        ]
        
        # Simple extraction
        keywords = []
        
        # Check for skill terms
        for term in skill_terms:
            if term.lower() in job_description.lower():
                # Find context
                pattern = re.compile(r'[^.!?]*\b' + re.escape(term) + r'\b[^.!?]*[.!?]', re.IGNORECASE)
                matches = pattern.findall(job_description)
                
                context = matches[0].strip() if matches else f"Keyword: {term}"
                
                # Determine if hard or soft skill
                soft_skills = ["leadership", "communication", "presentation", "management", 
                               "collaboration", "teamwork", "problem-solving"]
                skill_type = "soft skill" if any(s in term.lower() for s in soft_skills) else "hard skill"
                
                # Simple relevance scoring
                relevance = 7  # Default medium-high relevance
                if "required" in context.lower() or "must have" in context.lower():
                    relevance = 9
                elif "preferred" in context.lower() or "nice to have" in context.lower():
                    relevance = 5
                
                keywords.append({
                    "keyword": term,
                    "context": context,
                    "relevance_score": relevance,
                    "skill_type": skill_type
                })
        
        # Extract capitalized phrases (potential domain-specific terms)
        capitalized_pattern = re.compile(r'\b[A-Z][a-zA-Z]+(?: [A-Za-z]+){0,3}\b')
        capitalized_terms = capitalized_pattern.findall(job_description)
        
        for term in set(capitalized_terms):
            # Skip if already in keywords or too short
            if term.lower() in [k["keyword"].lower() for k in keywords] or len(term) < 3:
                continue
                
            # Find context
            pattern = re.compile(r'[^.!?]*\b' + re.escape(term) + r'\b[^.!?]*[.!?]')
            matches = pattern.findall(job_description)
            
            context = matches[0].strip() if matches else f"Keyword: {term}"
            
            keywords.append({
                "keyword": term,
                "context": context,
                "relevance_score": 6,  # Medium relevance for unknown terms
                "skill_type": "hard skill"  # Assume domain knowledge is hard skill
            })
        
        return keywords
    
    def _filter_quality_keywords(self, keywords: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter keywords for quality.
        
        Args:
            keywords: Raw extracted keywords
            
        Returns:
            list: Quality filtered keywords
        """
        # Generic terms to filter out
        generic_terms = {
            "experience", "skill", "ability", "knowledge", "capability", "qualification",
            "background", "expertise", "proficiency", "job", "work", "career", "position",
            "applicant", "candidate", "employer", "employee", "application", "resume"
        }
        
        # Apply quality filters
        filtered = []
        
        for kw in keywords:
            keyword = kw.get("keyword", "").strip()
            context = kw.get("context", "").strip()
            relevance = kw.get("relevance_score", 0)
            
            # Skip invalid entries
            if not keyword or not context:
                continue
                
            # Skip generic terms (case-insensitive)
            if keyword.lower() in generic_terms:
                continue
                
            # Length check
            if len(keyword) < 2 or len(keyword) > 50:
                continue
                
            # Minimum relevance
            if relevance < 4:
                continue
                
            # Context check
            if len(context) < 10:
                continue
                
            # Add to filtered list
            filtered.append(kw)
        
        # Sort by relevance score
        filtered.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return filtered
    
    def _calculate_statistics(self, keywords: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about the keywords.
        
        Args:
            keywords: Filtered keywords
            
        Returns:
            dict: Statistics about keywords
        """
        # Count by skill type
        hard_skills = sum(1 for k in keywords if k.get("skill_type") == "hard skill")
        soft_skills = sum(1 for k in keywords if k.get("skill_type") == "soft skill")
        
        # Count by relevance
        high_relevance = sum(1 for k in keywords if k.get("relevance_score", 0) >= 8)
        medium_relevance = sum(1 for k in keywords if 5 <= k.get("relevance_score", 0) < 8)
        low_relevance = sum(1 for k in keywords if k.get("relevance_score", 0) < 5)
        
        return {
            "total_keywords": len(keywords),
            "hard_skills": hard_skills,
            "soft_skills": soft_skills,
            "relevance": {
                "high": high_relevance,
                "medium": medium_relevance,
                "low": low_relevance
            }
        }


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract keywords from job description")
    parser.add_argument("--jd-file", type=str, help="Path to job description file")
    parser.add_argument("--jd-text", type=str, help="Job description text")
    parser.add_argument("--output", type=str, default="keywords.json", help="Output file path")
    
    args = parser.parse_args()
    
    # Get job description
    if args.jd_file:
        with open(args.jd_file, 'r', encoding='utf-8') as f:
            job_description = f.read()
    elif args.jd_text:
        job_description = args.jd_text
    else:
        parser.error("Either --jd-file or --jd-text is required")
    
    # Extract keywords
    extractor = KeywordExtractor()
    result = extractor.extract_keywords(job_description)
    
    # Print summary
    print(f"Extracted {result['statistics']['total_keywords']} keywords")
    print(f"Hard skills: {result['statistics']['hard_skills']}")
    print(f"Soft skills: {result['statistics']['soft_skills']}")
    print(f"High relevance: {result['statistics']['relevance']['high']}")
    
    # Save to file
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        
    print(f"Keywords saved to {args.output}")
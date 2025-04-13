"""
Simplified JSON to LaTeX converter for resume optimization pipeline.

This module takes the JSON output from our resume optimization and 
converts it to LaTeX format using the existing template.
"""

import os
import json
import subprocess
from typing import Dict, Any, Optional


def json_to_latex(resume_data: Dict[str, Any]) -> str:
    """
    Adapt our pipeline's JSON output to LaTeX format using existing classic_template.py functions.
    
    Args:
        resume_data: JSON output from the resume optimizer
        
    Returns:
        str: LaTeX content
    """
    # Import the classic template functions
    from classic_template import (
        fix_latex_special_chars,
        generate_header_section,
        generate_education_section,
        generate_experience_section,
        generate_projects_section,
        generate_certifications_section,
        generate_skills_section,
        generate_involvement_section,
        generate_awards_section,
        generate_latex_content
    )
    
    # Convert our JSON structure to the expected structure
    template_data = convert_to_template_format(resume_data)
    
    # Generate LaTeX content using the existing template
    latex_content = generate_latex_content(template_data)
    
    return latex_content


def convert_to_template_format(resume_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert our pipeline's JSON structure to the format expected by the LaTeX template.
    
    Args:
        resume_data: JSON output from the resume optimizer
        
    Returns:
        dict: Converted data format for the template
    """
    result = {}
    
    # Personal information
    personal_info = resume_data.get("Personal Information", {})
    result["name"] = personal_info.get("name", "")
    
    # Contact information
    result["contact"] = {
        "email": personal_info.get("email", ""),
        "phone": personal_info.get("phone", ""),
        "linkedin": personal_info.get("website/LinkedIn", "")
    }
    
    # Location
    result["location"] = personal_info.get("location", "")
    
    # Education
    education_list = []
    for edu in resume_data.get("Education", []):
        education_item = {
            "institution": edu.get("university", ""),
            "location": edu.get("location", ""),
            "degree": edu.get("degree", ""),
            "specialization": edu.get("specialization", ""),
            "dates": "",  # Will be built from start_date and end_date
            "start_date": edu.get("start_date", ""),
            "end_date": edu.get("end_date", ""),
            "gpa": edu.get("gpa", ""),
            "honors": edu.get("honors", ""),
            "relevant_coursework": edu.get("additional_info", "").split(",") if isinstance(edu.get("additional_info"), str) else []
        }
        education_list.append(education_item)
    result["education"] = education_list
    
    # Work Experience
    work_experience = []
    for exp in resume_data.get("Experience", []):
        responsibilities = exp.get("responsibilities/achievements", [])
        experience_item = {
            "company": exp.get("company", ""),
            "location": exp.get("location", ""),
            "position": exp.get("title", ""),
            "dates": "",  # Will be built from start_date and end_date
            "start_date": exp.get("start_date", ""),
            "end_date": exp.get("end_date", ""),
            "responsibilities": responsibilities
        }
        work_experience.append(experience_item)
    result["work_experience"] = work_experience
    
    # Projects
    projects = []
    for proj in resume_data.get("Projects", []):
        project_item = {
            "title": proj.get("title", ""),
            "description": proj.get("description", ""),
            "technologies": proj.get("technologies", [])
        }
        projects.append(project_item)
    result["projects"] = projects
    
    # Skills
    skills = {}
    tech_skills = resume_data.get("Skills", {}).get("Technical Skills", [])
    soft_skills = resume_data.get("Skills", {}).get("Soft Skills", [])
    
    if tech_skills:
        skills["Technical"] = tech_skills
    if soft_skills:
        skills["Soft"] = soft_skills
    result["skills"] = skills
    
    # Certifications
    certifications = []
    for cert in resume_data.get("Certifications/Awards", []):
        if isinstance(cert, dict):
            certifications.append(cert)
        else:
            certifications.append({"certification": cert})
    result["certifications"] = certifications
    
    # Volunteer Experience as Involvement
    extracurriculars = []
    for vol in resume_data.get("Volunteer Experience", []):
        if isinstance(vol, dict):
            extracurriculars.append({
                "organization": vol.get("organization", ""),
                "position": vol.get("role", "")
            })
        else:
            extracurriculars.append({
                "organization": "Volunteer",
                "position": vol
            })
    result["extracurriculars_certifications"] = extracurriculars
    
    return result


def convert_json_to_pdf(json_path: str, output_dir: str = "output", output_name: str = "enhanced_resume") -> Dict[str, str]:
    """
    Convert JSON file to LaTeX and PDF.
    
    Args:
        json_path: Path to the JSON file
        output_dir: Directory to save output files
        output_name: Base name for output files
        
    Returns:
        dict: Paths to output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        resume_data = json.load(f)
    
    # Generate LaTeX content
    latex_content = json_to_latex(resume_data)
    
    # Write LaTeX to file
    latex_path = os.path.join(output_dir, f"{output_name}.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_content)
    
    # Compile LaTeX to PDF
    pdf_path = os.path.join(output_dir, f"{output_name}.pdf")
    try:
        # Run pdflatex
        subprocess.run(
            ['pdflatex', '-output-directory', output_dir, latex_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Run it twice to resolve references (if any)
        subprocess.run(
            ['pdflatex', '-output-directory', output_dir, latex_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        print(f"PDF successfully generated: {pdf_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error compiling PDF: {e}")
        pdf_path = None
    except FileNotFoundError:
        print("pdflatex not found. Please install LaTeX (e.g., texlive)")
        pdf_path = None
    
    return {
        "latex": latex_path,
        "pdf": pdf_path
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert resume JSON to PDF")
    parser.add_argument("--json", required=True, help="Path to JSON file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--output-name", default="enhanced_resume", help="Base name for output files")
    
    args = parser.parse_args()
    
    convert_json_to_pdf(args.json, args.output_dir, args.output_name)
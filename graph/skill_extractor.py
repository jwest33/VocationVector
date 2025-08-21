#!/usr/bin/env python
"""Simple skill extractor for when LLM fails"""

import re
from typing import List, Dict, Set

# Comprehensive payroll and billing skills
PAYROLL_SKILLS = {
    # Payroll software/systems
    "ADP", "Workday", "Paychex", "QuickBooks", "NetSuite", "Ceridian",
    "Paylocity", "Gusto", "BambooHR", "UltiPro", "Kronos", "Paycom",
    "TriNet", "Zenefits", "Rippling", "Namely", "Justworks", "SAP",
    "Oracle", "PeopleSoft", "Sage", "Xero", "Wave", "FreshBooks",
    
    # Payroll processes
    "payroll processing", "payroll", "billing", "invoicing", "collections",
    "garnishments", "deductions", "benefits administration", "benefits",
    "401k", "retirement plans", "HSA", "FSA", "PTO", "time tracking",
    "overtime", "commission", "bonuses", "expense reimbursement",
    "multi-state payroll", "international payroll", "global payroll",
    "payroll reconciliation", "payroll auditing", "payroll reporting",
    
    # Compliance and tax
    "tax compliance", "payroll tax", "W-2", "W-4", "1099", "1040",
    "Form 941", "Form 940", "state tax", "local tax", "FICA", "FUTA",
    "SUTA", "SOX compliance", "GAAP", "IFRS", "labor laws", "FLSA",
    "ACA compliance", "COBRA", "HIPAA", "ERISA", "DOL compliance",
    
    # Financial/accounting
    "general ledger", "GL", "journal entries", "reconciliation",
    "financial reporting", "month-end close", "year-end close",
    "accounts payable", "accounts receivable", "AR", "AP",
    "budgeting", "forecasting", "financial analysis", "variance analysis",
    "audit", "internal controls", "financial statements",
    
    # Technical skills
    "Excel", "pivot tables", "VLOOKUP", "macros", "Power BI", "Tableau",
    "SQL", "data analysis", "reporting", "dashboards", "KPIs",
    "process improvement", "automation", "system implementation",
    
    # Management skills
    "team management", "leadership", "training", "mentoring",
    "project management", "vendor management", "client relations",
    "communication", "problem-solving", "attention to detail",
    "time management", "organization", "multitasking", "prioritization"
}

def extract_skills_from_text(text: str) -> List[Dict[str, any]]:
    """Extract skills from job text using keyword matching"""
    
    if not text:
        return []
    
    text_lower = text.lower()
    found_skills = set()
    
    # Check each skill
    for skill in PAYROLL_SKILLS:
        skill_lower = skill.lower()
        
        # Use word boundaries for better matching
        patterns = [
            r'\b' + re.escape(skill_lower) + r'\b',  # Exact word
            r'\b' + re.escape(skill_lower) + r's?\b',  # Plural
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower):
                found_skills.add(skill)
                break
    
    # Also extract any software mentioned after specific keywords
    software_patterns = [
        r'experience with ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        r'proficiency in ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        r'knowledge of ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
        r'using ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)',
    ]
    
    for pattern in software_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) > 2 and len(match) < 30:  # Reasonable length
                found_skills.add(match)
    
    # Convert to list of dicts for compatibility
    skills_list = []
    for skill in sorted(found_skills):
        skills_list.append({
            "skill_name": skill,
            "required_proficiency": "intermediate",
            "years_required": None,
            "is_mandatory": False,
            "evidence": ["Mentioned in job posting"],
            "confidence": 0.8
        })
    
    return skills_list


def enhance_template_with_extracted_skills(template: Dict, job_text: str) -> Dict:
    """Enhance a template with extracted skills if it has too few"""
    
    # Check if template needs enhancement
    current_skills = template.get("technical_skills", [])
    if len(current_skills) < 5:  # If less than 5 skills, extract more
        extracted_skills = extract_skills_from_text(job_text)
        
        # Merge with existing skills
        existing_names = {s.get("skill_name", s) for s in current_skills if isinstance(s, dict)}
        
        for skill in extracted_skills:
            if skill["skill_name"] not in existing_names:
                current_skills.append(skill)
        
        template["technical_skills"] = current_skills
        
    return template


if __name__ == "__main__":
    # Test with sample job text
    test_text = """
    Fractional Billing & Payroll Manager
    
    Requirements:
    - Proven experience as a Billing and Payroll Manager
    - Proficiency in accounting software and payroll systems
    - Experience with ADP, Workday, QuickBooks
    - Knowledge of tax compliance, W-2, 1099 processing
    - Strong Excel skills including pivot tables and VLOOKUP
    - Attention to detail and problem-solving skills
    """
    
    skills = extract_skills_from_text(test_text)
    print(f"Extracted {len(skills)} skills:")
    for skill in skills[:10]:
        print(f"  - {skill['skill_name']}")

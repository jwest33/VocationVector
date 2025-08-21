"""
Job Processing Node - Extracts structured information from job postings into universal template

This module processes raw job postings and extracts structured information with evidence
and confidence scores for each field, enabling bidirectional matching with resumes.
"""

from __future__ import annotations
import os
import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# LLM client
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Universal template
from graph.vocation_template import (
    VocationTemplate,
    SkillRequirement,
    ExperienceRequirement,
    EducationRequirement,
    LocationRequirement,
    CompensationRequirement,
    CultureFit,
    WorkArrangement,
    EmploymentType
)

# Validation
from pydantic import BaseModel, Field


# -------------------------
# Configuration
# -------------------------

class JobProcessingConfig:
    """Configuration for job processing"""
    
    # LLM settings
    DEFAULT_LLM_MODEL: str = "qwen3-4b-instruct-2507-f16"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 10000
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT: int = 240  # Increased from 120 to handle longer processing
    DEFAULT_BASE_URL: str = "http://localhost:8000/v1"
    
    # Processing settings
    MAX_JOB_TEXT_LENGTH: int = 100000  # Increased to handle full job postings
    EXTRACT_EVIDENCE: bool = True
    MIN_CONFIDENCE_THRESHOLD: float = 0.3
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'JobProcessingConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# -------------------------
# Data Models for LLM Response
# -------------------------

class JobExtraction(BaseModel):
    """Validated LLM response for job extraction"""
    title: str = Field(default="", max_length=500)
    company: str = Field(default="", max_length=500)
    
    # Skills
    technical_skills: List[Dict[str, Any]] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    tools_technologies: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    
    # Experience
    years_experience_required: Optional[float] = None
    experience_requirements: List[str] = Field(default_factory=list)
    management_experience_required: Optional[float] = None
    
    # Education
    education_requirements: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Location & Remote
    location: Optional[str] = None
    remote_policy: Optional[str] = None
    relocation_assistance: Optional[bool] = None
    visa_sponsorship: Optional[bool] = None
    
    # Compensation
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: str = Field(default="USD")
    benefits: List[str] = Field(default_factory=list)
    equity: Optional[str] = None
    bonus: Optional[str] = None
    
    # Employment details
    employment_type: Optional[str] = None
    start_date: Optional[str] = None
    
    # Responsibilities
    responsibilities: List[str] = Field(default_factory=list)
    
    # Company culture
    company_values: List[str] = Field(default_factory=list)
    team_size: Optional[str] = None
    growth_opportunities: Optional[str] = None
    
    # Summary
    job_summary: str = Field(default="", max_length=1000)
    
    # Evidence for key requirements
    evidence: Dict[str, List[str]] = Field(default_factory=dict)


# -------------------------
# LLM Processing
# -------------------------

class JobLLMProcessor:
    """Process job postings using LLM to extract structured information"""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        config: JobProcessingConfig = None
    ):
        self.config = config or JobProcessingConfig()
        self.model = model or os.getenv("LLM_MODEL", self.config.DEFAULT_LLM_MODEL)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", self.config.DEFAULT_BASE_URL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        logging.info(f"Initialized Job LLM processor with model: {self.model} at {self.base_url}")
    
    def _analyze_job_domain(self, job_text: str) -> Dict[str, Any]:
        """First pass: Analyze job domain and type"""
        
        # Take a sample for analysis
        sample = job_text
        
        system_prompt = "Identify job domain and key skills. Return JSON only."
        
        user_prompt = f"""Analyze this job posting:

{sample}

Return JSON:
{{
  "domain": "payroll|tech|sales|hr|finance|other",
  "level": "entry|mid|senior|manager|director",
  "key_skills_mentioned": ["skill1", "skill2"],
  "has_technical_requirements": true,
  "has_certifications": false,
  "industry": "technology|healthcare|finance|retail|other"
}}"""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            analysis = self._parse_llm_response(response)
            logging.info(f"Job analysis: domain={analysis.get('domain')}, level={analysis.get('level')}")
            return analysis
        except Exception as e:
            logging.error(f"Job analysis failed: {e}")
            raise RuntimeError(f"LLM extraction failed for job domain analysis: {e}")

    def _build_domain_specific_prompt(self, job_text: str, analysis: Dict[str, Any]) -> Tuple[str, str]:
        """Build domain-specific extraction prompt based on analysis"""
        
        domain = analysis.get("domain", "general")
        level = analysis.get("level", "mid")
        
        # Domain-specific skill hints
        skill_hints = {
            "payroll": [
                "ADP", "Workday", "Paychex", "QuickBooks", "NetSuite", "Ceridian",
                "Paylocity", "Gusto", "BambooHR", "UltiPro", "Kronos",
                "Excel", "GAAP", "SOX compliance", "tax compliance", "garnishments",
                "multi-state payroll", "international payroll", "benefits administration",
                "401k", "W-2", "1099", "payroll accounting", "GL reconciliation"
            ],
            "tech": [
                "Python", "Java", "JavaScript", "TypeScript", "C++", "Go", "Rust",
                "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring",
                "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
                "SQL", "NoSQL", "MongoDB", "PostgreSQL", "Redis", "Kafka",
                "CI/CD", "Jenkins", "GitLab", "GitHub Actions", "Agile", "Scrum"
            ],
            "finance": [
                "QuickBooks", "SAP", "Oracle", "NetSuite", "Excel", "Power BI",
                "Tableau", "SQL", "GAAP", "IFRS", "SOX", "financial modeling",
                "budgeting", "forecasting", "audit", "tax", "CPA", "CFA"
            ],
            "hr": [
                "Workday", "BambooHR", "ADP", "SuccessFactors", "Taleo", "iCIMS",
                "recruiting", "onboarding", "performance management", "HRIS",
                "employee relations", "compensation", "benefits", "SHRM", "PHR"
            ]
        }
        
        relevant_skills = skill_hints.get(domain, [])
        skills_hint = f"Look especially for these {domain} skills: {', '.join(relevant_skills[:15])}" if relevant_skills else ""
        
        system_prompt = f"Extract {domain} job requirements. Focus on skills, tools, and qualifications. Return complete JSON."
        
        # Truncate if needed
        # Don't truncate job text - use full content
        # if len(job_text) > self.config.MAX_JOB_TEXT_LENGTH:
        #     job_text = job_text[:self.config.MAX_JOB_TEXT_LENGTH]
        
        user_prompt = f"""Extract ALL skills and requirements from this job posting.

Job Posting:
{job_text}

INSTRUCTIONS FOR SKILLS EXTRACTION:

1. TECHNICAL SKILLS: Programming languages, frameworks, methodologies, technical concepts
   Examples: Python, Java, React, Agile, Machine Learning, API Development, Database Design

2. TOOLS/TECHNOLOGIES: Specific software, platforms, tools, systems, cloud services  
   Examples: AWS, Docker, Kubernetes, Jira, Git, MySQL, MongoDB, Salesforce, Excel, QuickBooks

3. SOFT SKILLS: Interpersonal and professional skills
   Examples: Communication, Leadership, Problem-solving, Team collaboration, Time management

{skills_hint}

Return JSON with these exact fields:
{{
  "title": "exact job title from posting",
  "company": "company name",
  "technical_skills": [
    "ONLY list programming languages, frameworks, and technical concepts EXPLICITLY mentioned in the job"
  ],
  "tools_technologies": [
    "ONLY list specific tools, software, platforms, and systems EXPLICITLY mentioned in the job"
  ],
  "soft_skills": [
    "ONLY list interpersonal and professional skills EXPLICITLY mentioned in the job"
  ],
  "certifications": ["List any certifications mentioned"],
  "years_experience_required": 5,
  "salary_min": 60000,
  "salary_max": 80000,
  "location": "location",
  "remote_policy": "remote or hybrid or onsite"
}}

IMPORTANT RULES:
- Keep skills as simple strings (not objects)
- Don't duplicate items between technical_skills and tools_technologies
- A skill goes in technical_skills if it's a language/framework/methodology
- A skill goes in tools_technologies if it's a specific product/tool/platform
- Extract ALL mentioned skills, don't limit the list

SALARY RULES:
- "$85k" → 85000
- "$40/hour" → 83200 (2080 hours/year)
- "$4,000/month" → 48000
- If no salary, use null"""
        
        return system_prompt, user_prompt
    
    def _build_extraction_prompt(self, job_text: str) -> Tuple[str, str]:
        """Two-pass approach: analyze then extract with domain-specific prompt"""
        
        # First pass: analyze job domain
        analysis = self._analyze_job_domain(job_text)
        
        # Second pass: build domain-specific prompt
        return self._build_domain_specific_prompt(job_text, analysis)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM with retry logic"""
        import time
        import requests
        
        # First check if server is reachable
        max_retries = 3
        for retry in range(max_retries):
            try:
                # Quick health check with short timeout
                health_url = f"{self.base_url.replace('/v1', '')}/health"
                response = requests.get(health_url, timeout=1)
                if response.status_code == 200:
                    break
            except:
                if retry < max_retries - 1:
                    logging.warning(f"LLM server not reachable at {self.base_url}, retrying in {2 ** retry} seconds...")
                    time.sleep(2 ** retry)
                else:
                    # Try without /v1 suffix as fallback
                    try:
                        alt_base_url = self.base_url.replace('/v1', '')
                        response = requests.get(f"{alt_base_url}/health", timeout=1)
                        if response.status_code == 200:
                            self.base_url = alt_base_url
                            self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)
                            logging.info(f"Using alternate base URL: {self.base_url}")
                    except:
                        pass
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
                timeout=self.config.LLM_TIMEOUT,
                extra_body={
                    "top_k": 40,
                    "top_p": 0.95,
                    "repeat_penalty": 1.1,
                    "seed": 42,
                    "stop": ["\n\n}", "}\n\n", "```"],
                    "stream": False
                }
            )
            
            content = response.choices[0].message.content
            logging.info(f"LLM raw response (first 500 chars): {content[:500]}")
            return content
            
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            # Add more context to the error
            if "Connection" in str(e):
                logging.error(f"Connection error - server may not be running at {self.base_url}")
                logging.error("Ensure the LLM server is started before running the pipeline")
            raise
    
    def _parse_salary(self, salary_value) -> Optional[float]:
        """Parse salary strings to numeric values"""
        if salary_value is None:
            return None
            
        # If already numeric, return it (but check for NaN)
        if isinstance(salary_value, (int, float)):
            if isinstance(salary_value, float) and (salary_value != salary_value or salary_value == float('inf') or salary_value == float('-inf')):
                return None
            return float(salary_value)
        
        # Parse string formats
        if isinstance(salary_value, str):
            salary_str = salary_value.strip().lower()
            
            # Remove currency symbols and commas
            salary_str = salary_str.replace('$', '').replace(',', '').replace(' ', '')
            
            # Handle 'k' notation (e.g., "85k" -> 85000)
            if 'k' in salary_str:
                try:
                    num = float(salary_str.replace('k', ''))
                    return num * 1000
                except:
                    pass
            
            # Handle hourly rates (e.g., "40/hour" -> annual)
            if 'hour' in salary_str or '/hr' in salary_str:
                try:
                    # Extract number
                    num_match = re.search(r'(\d+(?:\.\d+)?)', salary_str)
                    if num_match:
                        hourly = float(num_match.group(1))
                        return hourly * 2080  # Standard work hours per year
                except:
                    pass
            
            # Handle monthly rates
            if 'month' in salary_str or '/mo' in salary_str:
                try:
                    num_match = re.search(r'(\d+(?:\.\d+)?)', salary_str)
                    if num_match:
                        monthly = float(num_match.group(1))
                        return monthly * 12
                except:
                    pass
            
            # Try to parse as regular number
            try:
                return float(salary_str)
            except:
                pass
        
        return None
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response with error handling"""
        
        if not response or response.strip() == "":
            logging.warning("Empty LLM response received")
            return {}
        
        response = response.strip()
        
        # Check if response was cut off
        if response and not response.endswith('}'):
            logging.warning(f"Response appears truncated: {response[-50:]}")
            # Try to complete the JSON
            if '"' in response:
                response += '"}'
            else:
                response += '}'
        
        # Remove markdown if present
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            parts = response.split("```")
            if len(parts) >= 2:
                response = parts[1]
        
        response = response.strip()
        
        try:
            data = json.loads(response)
            
            # Parse salary fields if they're strings
            if 'salary_min' in data:
                data['salary_min'] = self._parse_salary(data['salary_min'])
            if 'salary_max' in data:
                data['salary_max'] = self._parse_salary(data['salary_max'])
            
            return data
            
        except json.JSONDecodeError as e:
            logging.debug(f"JSON parse failed: {e}")
            
            # Try to extract JSON pattern
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}[^{}]*)*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                try:
                    json_str = json_match.group(0)
                    # Clean up common issues
                    json_str = re.sub(r',\s*]', ']', json_str)
                    json_str = re.sub(r',\s*}', '}', json_str)
                    data = json.loads(json_str)
                    
                    # Parse salary fields if they're strings
                    if 'salary_min' in data:
                        data['salary_min'] = self._parse_salary(data['salary_min'])
                    if 'salary_max' in data:
                        data['salary_max'] = self._parse_salary(data['salary_max'])
                    
                    return data
                except:
                    pass
            
            # Return minimal structure
            logging.warning("Failed to parse LLM response, returning minimal structure")
            return {
                "title": "",
                "company": "",
                "technical_skills": [],
                "years_experience_required": None,
                "responsibilities": []
            }
    
    def _extract_basic_info(self, job_text: str) -> Dict[str, Any]:
        """First pass: Extract basic job information"""
        system_prompt = "Extract basic job information. Return JSON only."
        
        # Take first 2000 chars for basic info
        sample = job_text[:2000] if len(job_text) > 2000 else job_text
        
        user_prompt = f"""Extract from this job posting:
{sample}

Return JSON:
{{
  "title": "exact job title",
  "company": "company name",
  "location": "location",
  "remote_policy": "remote/hybrid/onsite",
  "employment_type": "full-time/part-time/contract",
  "salary_min": null,
  "salary_max": null,
  "years_experience_required": 5
}}

SALARY RULES: Convert to annual numbers. $85k=85000, $40/hour=83200"""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logging.error(f"Basic info extraction failed: {e}")
            raise RuntimeError(f"LLM extraction failed for basic info: {e}")
    
    def _extract_skills(self, job_text: str, domain: str = "general") -> Dict[str, Any]:
        """Second pass: Extract all skills and technologies"""
        system_prompt = "Extract ALL skills from job posting. Return JSON only."
        
        user_prompt = f"""Extract skills from this job:
{job_text}

CATEGORIES:
1. TECHNICAL SKILLS: Programming languages, frameworks, methodologies
   Examples: Python, Java, React, REST APIs, Agile, Machine Learning

2. TOOLS/TECHNOLOGIES: Specific software, platforms, systems
   Examples: AWS, Docker, Excel, QuickBooks, Jira, PostgreSQL

3. SOFT SKILLS: Interpersonal skills
   Examples: Communication, Leadership, Problem-solving

Return JSON:
{{
  "technical_skills": ["list only skills EXPLICITLY mentioned in the job"],
  "tools_technologies": ["list only tools EXPLICITLY mentioned in the job"],
  "soft_skills": ["list only soft skills EXPLICITLY mentioned in the job"],
  "certifications": ["list only certifications EXPLICITLY mentioned in the job"]
}}

CRITICAL RULES:
- ONLY extract skills that are EXPLICITLY mentioned in the job posting
- DO NOT add skills that seem related but aren't mentioned
- DO NOT include example skills unless they appear in the actual job text
- If a category has no items mentioned, return an empty array []
- Keep as simple strings, not objects"""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logging.error(f"Skills extraction failed: {e}")
            raise RuntimeError(f"LLM extraction failed for skills: {e}")
    
    def _extract_requirements(self, job_text: str) -> Dict[str, Any]:
        """Third pass: Extract requirements and responsibilities"""
        system_prompt = "Extract job requirements and responsibilities. Return JSON only."
        
        user_prompt = f"""Extract from this job:
{job_text}

Return JSON:
{{
  "responsibilities": [
    "Main job duty 1",
    "Main job duty 2"
  ],
  "experience_requirements": [
    "5+ years experience in X",
    "Experience with Y"
  ],
  "education_requirements": [
    {{"degree": "Bachelor's", "field": "Computer Science", "required": true}}
  ],
  "benefits": ["Health insurance", "401k"],
  "job_summary": "One paragraph summary of the role"
}}"""
        
        try:
            response = self._call_llm(system_prompt, user_prompt)
            return self._parse_llm_response(response)
        except Exception as e:
            logging.error(f"Requirements extraction failed: {e}")
            raise RuntimeError(f"LLM extraction failed for requirements: {e}")
    
    def extract_job_info(self, job_text: str) -> JobExtraction:
        """Extract structured information from job posting using multi-pass approach"""
        try:
            logging.info("Extracting job information using multi-pass approach...")
            
            # Pass 1: Basic information
            basic_info = self._extract_basic_info(job_text)
            logging.info(f"Pass 1 - Basic info extracted: title={basic_info.get('title', 'Unknown')}")
            
            # Pass 2: Analyze domain for better skill extraction
            domain_analysis = self._analyze_job_domain(job_text)
            domain = domain_analysis.get("domain", "general")
            
            # Pass 3: Extract skills with domain context
            skills_info = self._extract_skills(job_text, domain)
            logging.info(f"Pass 2 - Skills extracted: {len(skills_info.get('technical_skills', []))} technical, {len(skills_info.get('tools_technologies', []))} tools")
            
            # Pass 4: Extract requirements and responsibilities
            requirements_info = self._extract_requirements(job_text)
            logging.info(f"Pass 3 - Requirements extracted: {len(requirements_info.get('responsibilities', []))} responsibilities")
            
            # Combine all extracted data
            combined_data = {
                **basic_info,
                **skills_info,
                **requirements_info
            }
            
            # Log what we got
            logging.info(f"Combined data keys: {list(combined_data.keys())}")
            logging.info(f"Technical skills: {combined_data.get('technical_skills', [])}")
            
            # Process technical skills - now they're simple strings from multi-pass
            technical_skills = []
            for skill in combined_data.get("technical_skills", []):
                if isinstance(skill, str) and skill.strip():
                    # Convert string to dict format for compatibility
                    technical_skills.append({
                        "skill": skill.strip(),
                        "required": True,
                        "years": None,
                        "evidence": [f"{skill} mentioned in job posting"]
                    })
            
            # Ensure all fields have proper defaults
            extraction_data = {
                "title": combined_data.get("title", ""),
                "company": combined_data.get("company", ""),
                "technical_skills": technical_skills,
                "soft_skills": combined_data.get("soft_skills", []),
                "tools_technologies": combined_data.get("tools_technologies", []),
                "certifications": combined_data.get("certifications", []),
                "years_experience_required": combined_data.get("years_experience_required"),
                "experience_requirements": combined_data.get("experience_requirements", []),
                "management_experience_required": combined_data.get("management_experience_required"),
                "education_requirements": combined_data.get("education_requirements", []),
                "location": combined_data.get("location"),
                "remote_policy": combined_data.get("remote_policy"),
                "relocation_assistance": combined_data.get("relocation_assistance"),
                "visa_sponsorship": combined_data.get("visa_sponsorship"),
                "salary_min": combined_data.get("salary_min"),
                "salary_max": combined_data.get("salary_max"),
                "salary_currency": combined_data.get("salary_currency", "USD"),
                "benefits": combined_data.get("benefits", []),
                "equity": combined_data.get("equity"),
                "bonus": combined_data.get("bonus"),
                "employment_type": combined_data.get("employment_type"),
                "start_date": combined_data.get("start_date"),
                "responsibilities": combined_data.get("responsibilities", []),
                "company_values": combined_data.get("company_values", []),
                "team_size": combined_data.get("team_size"),
                "growth_opportunities": combined_data.get("growth_opportunities"),
                "job_summary": combined_data.get("job_summary", ""),
                "evidence": combined_data.get("evidence", {})
            }
            
            logging.info(f"Extraction complete - found {len(extraction_data['technical_skills'])} technical skills")
            
            return JobExtraction(**extraction_data)
            
        except Exception as e:
            logging.error(f"Job extraction failed: {e}")
            raise RuntimeError(f"Failed to extract job information: {e}")

def convert_to_vocation_template(
    extraction: JobExtraction,
    job_id: str,
    original_text: str = ""
) -> VocationTemplate:
    """Convert extracted job information to universal template"""
    
    # Use extracted title or fallback to regex extraction if empty
    title = extraction.title
    if not title or title == "" or title == "Unknown Position":
        # Try to extract from original text
        if original_text:
            title = _extract_job_title_fallback(original_text)
        else:
            title = "Unknown Position"
    
    template = VocationTemplate(
        source_type="job_posting",
        source_id=job_id,
        title=title,
        summary=extraction.job_summary
    )
    
    # Convert technical skills with evidence
    for skill_data in extraction.technical_skills:
        if isinstance(skill_data, dict):
            skill_name = skill_data.get("skill", "")
            if skill_name:  # Only add if skill name exists
                years = skill_data.get("years")
                skill_req = SkillRequirement(
                    skill_name=skill_name,
                    required_proficiency="advanced" if years and years >= 3 else "intermediate",
                    years_required=years,
                    is_mandatory=skill_data.get("required", True),
                    evidence=skill_data.get("evidence", []),
                    confidence=0.8 if skill_data.get("evidence") else 0.5
                )
                template.technical_skills.append(skill_req)
    
    # If no technical skills found, try to extract from the text directly
    if not template.technical_skills and original_text:
        # Common payroll/business systems and skills
        common_skills = [
            "ADP", "Workday", "QuickBooks", "Paychex", "NetSuite", "SAP",
            "Excel", "PowerPoint", "Word", "Outlook",
            "GAAP", "SOX", "SQL", "Python", "Java", "C++",
            "AWS", "Azure", "Docker", "Kubernetes",
            "Salesforce", "HubSpot", "Tableau", "Power BI"
        ]
        
        text_lower = original_text.lower()
        for skill in common_skills:
            if skill.lower() in text_lower:
                template.technical_skills.append(SkillRequirement(
                    skill_name=skill,
                    required_proficiency="intermediate",
                    is_mandatory=False,
                    evidence=["Mentioned in job posting"],
                    confidence=0.6
                ))
    
    # Add soft skills
    for skill in extraction.soft_skills:
        template.soft_skills.append(SkillRequirement(
            skill_name=skill,
            required_proficiency="intermediate",
            is_mandatory=False,
            confidence=0.7
        ))
    
    # Add tools and technologies
    for tool in extraction.tools_technologies:
        template.tools_technologies.append(SkillRequirement(
            skill_name=tool,
            required_proficiency="intermediate",
            is_mandatory=False,
            confidence=0.7
        ))
    
    # Add certifications
    template.certifications = extraction.certifications
    
    # Add experience requirements
    if extraction.years_experience_required:
        template.total_years_experience = extraction.years_experience_required
        template.experience_requirements.append(ExperienceRequirement(
            experience_type="general",
            years_required=extraction.years_experience_required,
            description=f"{extraction.years_experience_required}+ years experience required",
            is_mandatory=True,
            evidence=extraction.evidence.get("experience", []),
            confidence=0.8
        ))
    
    if extraction.management_experience_required:
        template.management_experience = extraction.management_experience_required
        template.experience_requirements.append(ExperienceRequirement(
            experience_type="management",
            years_required=extraction.management_experience_required,
            description=f"{extraction.management_experience_required}+ years management experience",
            is_mandatory=True,
            evidence=extraction.evidence.get("experience", []),
            confidence=0.8
        ))
    
    # Add education requirements
    for edu_data in extraction.education_requirements:
        if isinstance(edu_data, dict):
            edu_req = EducationRequirement(
                degree_level=edu_data.get("degree", ""),
                field_of_study=edu_data.get("field"),
                is_mandatory=edu_data.get("required", True),
                evidence=extraction.evidence.get("education", []),
                confidence=0.8
            )
            template.education_requirements.append(edu_req)
    
    # Set location requirements
    template.location = LocationRequirement(
        preferred_locations=[extraction.location] if extraction.location else [],
        work_arrangement=_parse_work_arrangement(extraction.remote_policy),
        relocation_willing=bool(extraction.relocation_assistance),
        visa_sponsorship_needed=bool(extraction.visa_sponsorship)
    )
    
    # Set compensation requirements
    template.compensation = CompensationRequirement(
        minimum_salary=extraction.salary_min,
        maximum_salary=extraction.salary_max,
        currency=extraction.salary_currency,
        benefits_required=extraction.benefits,
        equity_expectation=extraction.equity,
        bonus_structure=extraction.bonus
    )
    
    # Set employment type
    template.employment_type = _parse_employment_type(extraction.employment_type)
    template.start_date = extraction.start_date
    
    # Set culture fit
    template.culture_fit = CultureFit(
        company_values=extraction.company_values,
        team_size_preference=extraction.team_size,
        career_growth_importance=extraction.growth_opportunities
    )
    
    # Add responsibilities
    template.key_responsibilities = extraction.responsibilities
    
    # Set metadata
    template.metadata = {
        "company": extraction.company,
        "original_text_length": len(original_text),
        "extraction_evidence": extraction.evidence
    }
    template.extraction_confidence = 0.75  # Base confidence, can be improved with validation
    template.processed_at = datetime.now().isoformat()
    
    return template


def _parse_work_arrangement(remote_policy: Optional[str]) -> Optional[WorkArrangement]:
    """Parse work arrangement from remote policy string"""
    if not remote_policy:
        return None
    
    policy_lower = remote_policy.lower()
    if "remote" in policy_lower and "hybrid" not in policy_lower:
        return WorkArrangement.REMOTE
    elif "hybrid" in policy_lower:
        return WorkArrangement.HYBRID
    elif "onsite" in policy_lower or "office" in policy_lower:
        return WorkArrangement.ONSITE
    else:
        return WorkArrangement.FLEXIBLE


def _parse_employment_type(employment_type: Optional[str]) -> Optional[EmploymentType]:
    """Parse employment type from string"""
    if not employment_type:
        return None
    
    type_lower = employment_type.lower()
    if "full" in type_lower:
        return EmploymentType.FULL_TIME
    elif "part" in type_lower:
        return EmploymentType.PART_TIME
    elif "contract" in type_lower:
        return EmploymentType.CONTRACT
    elif "freelance" in type_lower:
        return EmploymentType.FREELANCE
    elif "intern" in type_lower:
        return EmploymentType.INTERNSHIP
    else:
        return EmploymentType.FULL_TIME  # Default

class JobProcessingPipeline:
    """Main pipeline for processing job postings"""
    
    def __init__(
        self,
        config: JobProcessingConfig = None,
        llm_processor: JobLLMProcessor = None
    ):
        self.config = config or JobProcessingConfig()
        self.llm_processor = llm_processor or JobLLMProcessor(config=self.config)
    
    def process_job(
        self,
        job_data: Dict[str, Any],
        verbose: bool = False
    ) -> Tuple[VocationTemplate, Dict[str, Any]]:
        """Process a job posting into universal template"""
        
        diagnostics = {
            "job_id": job_data.get("id", "unknown"),
            "errors": []
        }
        
        try:
            # Extract text from job data
            job_text = job_data.get("text", job_data.get("full_text", ""))
            if not job_text:
                raise ValueError("No job text found")
            
            diagnostics["text_length"] = len(job_text)
            
            # Process with LLM
            if verbose:
                logging.info("Processing job with LLM...")
            extraction = self.llm_processor.extract_job_info(job_text)
            
            # Convert to universal template
            job_id = str(job_data.get("id", f"job_{datetime.now().timestamp()}"))
            template = convert_to_vocation_template(extraction, job_id, job_text)
            
            # Override with any directly available data
            if job_data.get("title"):
                template.title = job_data["title"]
            if job_data.get("company"):
                template.metadata["company"] = job_data["company"]
            
            diagnostics["status"] = "success"
            diagnostics["skills_extracted"] = len(template.technical_skills)
            diagnostics["requirements_extracted"] = len(template.experience_requirements)
            
            return template, diagnostics
            
        except Exception as e:
            logging.error(f"Job processing failed: {e}")
            diagnostics["status"] = "error"
            diagnostics["errors"].append(str(e))
            
            # Try to extract title from text using regex patterns as fallback
            title = job_data.get("title", "")
            if not title or title == "Unknown Position":
                job_text = job_data.get("text", "")
                if job_text:
                    title = _extract_job_title_fallback(job_text)
                else:
                    title = "Unknown Position"
            
            # Return minimal template
            return VocationTemplate(
                source_type="job_posting",
                source_id=str(job_data.get("id", "unknown")),
                title=title,
                summary=""
            ), diagnostics
    
    def process_jobs_batch(
        self,
        jobs: List[Dict[str, Any]],
        verbose: bool = False
    ) -> Tuple[List[VocationTemplate], Dict[str, Any]]:
        """Process multiple job postings"""
        
        templates = []
        diagnostics = {
            "jobs_processed": 0,
            "jobs_failed": 0,
            "errors": []
        }
        
        for job_data in jobs:
            if verbose:
                logging.info(f"Processing job {job_data.get('job_id', job_data.get('job_index', 'unknown'))}")
            
            try:
                template, job_diag = self.process_job(job_data, verbose)
                templates.append(template)
                diagnostics["jobs_processed"] += 1
            except Exception as e:
                logging.error(f"Failed to process job: {e}")
                diagnostics["jobs_failed"] += 1
                diagnostics["errors"].append(str(e))
        
        return templates, diagnostics


def _extract_job_title_fallback(job_text: str) -> str:
    """Extract job title from text using regex patterns when LLM fails"""
    import re
    
    # Handle concatenated text (when everything is on one line)
    if '\n' not in job_text or job_text.count('\n') < 3:
            # Common job title keywords
            title_keywords = [
                'specialist', 'manager', 'engineer', 'developer', 'analyst',
                'coordinator', 'director', 'associate', 'consultant', 'administrator',
                'technician', 'designer', 'architect', 'lead', 'senior', 'junior',
                'supervisor', 'executive', 'officer', 'assistant', 'accountant',
                'representative', 'agent', 'advisor', 'expert', 'professional',
                'strategist', 'scientist', 'researcher', 'trainer', 'instructor',
                'payroll', 'billing', 'fractional'
            ]
            
            # Handle pattern where company name appears twice
            first_200_chars = job_text[:200] if len(job_text) > 200 else job_text
            words = first_200_chars.split()
            
            if len(words) >= 4:
                # Skip single letter prefixes
                if len(words[0]) == 1 and len(words) > 1:
                    words = words[1:]
                
                # Check for pattern: [Company] [Title] [Company] • [Location]
                for company_word_count in [3, 2, 1]:
                    if len(words) > company_word_count + 2:
                        potential_company = ' '.join(words[:company_word_count])
                        remaining_text = ' '.join(words[company_word_count:])
                        
                        # Check if company name appears again
                        if potential_company.lower() in remaining_text.lower():
                            company_second_pos = remaining_text.lower().find(potential_company.lower())
                            if company_second_pos > 0:
                                title_section = remaining_text[:company_second_pos].strip()
                                
                                # Clean up the title
                                title_section = re.sub(r'\s*-\s*Fully\s+Remote.*', '', title_section, flags=re.IGNORECASE)
                                title_section = re.sub(r'\s*-\s*Remote.*', '', title_section, flags=re.IGNORECASE)
                                title_section = re.sub(r'\s*-\s*US\s*$', '', title_section, flags=re.IGNORECASE)
                                title_section = re.sub(r'\s*\([^)]*\)\s*$', '', title_section)
                                title_section = title_section.strip(' -')
                                
                                if title_section and 2 <= len(title_section.split()) <= 8:
                                    title_lower = title_section.lower()
                                    if any(keyword in title_lower for keyword in title_keywords):
                                        return title_section.strip()
        
    return "Unknown Position"


def _extract_company_fallback(job_text: str) -> str:
    """Extract company name from text using regex patterns when LLM fails"""
    import re
    
    # Handle concatenated text
    if '\n' not in job_text or job_text.count('\n') < 3:
        first_200_chars = job_text[:200] if len(job_text) > 200 else job_text
        
        # For pattern like "E Everything To Gain Fractional Billing & Payroll Manager - US Everything To Gain • Remote, OR"
        # The company name appears twice, once at the beginning and once after the title
        
        # Try to find repeated company name pattern
        words = first_200_chars.split()
        if len(words) >= 4:
            # Skip single letter prefixes
            start_idx = 0
            if len(words[0]) == 1 and len(words) > 1:
                start_idx = 1
                words = words[1:]
            
            # Check for pattern where company name repeats
            for company_word_count in [3, 2, 1]:
                if len(words) > company_word_count + 2:
                    potential_company = ' '.join(words[:company_word_count])
                    remaining_text = ' '.join(words[company_word_count:])
                    
                    # Check if company name appears again
                    if potential_company.lower() in remaining_text.lower():
                        return potential_company.strip()
        
        # If no repeated pattern, try to extract from bullet separator
        if '•' in first_200_chars:
            parts = first_200_chars.split('•')
            if len(parts) >= 2:
                # Second part often contains location, so company might be in first part
                # Try to extract company from the end of the first part
                first_part = parts[0].strip()
                words = first_part.split()
                
                # Look for company indicators
                for i in range(len(words) - 1, -1, -1):
                    word = words[i]
                    # Check if this could be start of company name
                    if word[0].isupper() and i > 0:
                        # Check if previous words look like job title keywords
                        prev_word = words[i-1].lower()
                        if prev_word in ['manager', 'specialist', 'engineer', 'developer', 'analyst', 'director']:
                            # Company likely starts after the title
                            potential_company = ' '.join(words[i:])
                            if potential_company != first_part:  # Make sure we didn't take everything
                                return potential_company.strip()
    
    # For multi-line text
    lines = [line.strip() for line in job_text.split('\n') if line.strip()]
    
    # Often company is in second or third line
    for line in lines[1:5]:
        if line and len(line) < 100:
            # Skip lines that are clearly not company names
            if not any(skip in line.lower() for skip in ['apply', 'via', 'ago', 'full-time', 'part-time', 'http', 'www']):
                # Check if it looks like a company name
                if line[0].isupper() and len(line.split()) <= 5:
                    return line
    
    return "Unknown Company"


def job_processing_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible node for job processing.
    
    Expected state keys:
        - raw_jobs (list): List of raw job dictionaries
        - config (dict, optional): Configuration overrides
        - verbose (bool, optional): Enable verbose logging
    
    Returns state with:
        - processed_jobs: List of VocationTemplate dictionaries
        - job_templates: List of templates for matching
        - diagnostics: Processing diagnostics
    """
    
    # Set up logging
    if state.get("verbose", False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Get configuration
    config = JobProcessingConfig()
    if "config" in state and isinstance(state["config"], dict):
        config = JobProcessingConfig.from_dict(state["config"])
    
    # Initialize processor
    llm_processor = None
    if "llm_processor" in state:
        llm_processor = state["llm_processor"]
    
    pipeline = JobProcessingPipeline(
        config=config,
        llm_processor=llm_processor
    )
    
    try:
        # Get jobs to process
        raw_jobs = state.get("raw_jobs", [])
        
        if not raw_jobs:
            logging.warning("No jobs to process")
            state["processed_jobs"] = []
            state["job_templates"] = []
            return state
        
        logging.info(f"Processing {len(raw_jobs)} jobs")
        
        # Process jobs
        templates, diagnostics = pipeline.process_jobs_batch(
            raw_jobs,
            verbose=state.get("verbose", False)
        )
        
        # Convert templates to dictionaries
        processed_jobs = []
        for template in templates:
            # Convert to dictionary while handling enums
            job_dict = {
                "source_type": template.source_type,
                "source_id": template.source_id,
                "title": template.title,
                "summary": template.summary,
                "technical_skills": [asdict(s) for s in template.technical_skills],
                "soft_skills": [asdict(s) for s in template.soft_skills],
                "certifications": template.certifications,
                "tools_technologies": [asdict(s) for s in template.tools_technologies],
                "experience_requirements": [asdict(e) for e in template.experience_requirements],
                "total_years_experience": template.total_years_experience,
                "management_experience": template.management_experience,
                "education_requirements": [asdict(e) for e in template.education_requirements],
                "location": asdict(template.location),
                "compensation": asdict(template.compensation),
                "employment_type": template.employment_type.value if template.employment_type else None,
                "start_date": template.start_date,
                "culture_fit": asdict(template.culture_fit),
                "key_responsibilities": template.key_responsibilities,
                "achievements": template.achievements,
                "metadata": template.metadata,
                "extraction_confidence": template.extraction_confidence,
                "processed_at": template.processed_at
            }
            
            # Fix work arrangement enum
            if template.location.work_arrangement:
                job_dict["location"]["work_arrangement"] = template.location.work_arrangement.value
            
            processed_jobs.append(job_dict)
        
        # Update state
        state["processed_jobs"] = processed_jobs
        state["job_templates"] = templates  # Keep original template objects for matching
        
        if "diagnostics" not in state:
            state["diagnostics"] = {}
        state["diagnostics"]["job_processing"] = diagnostics
        
    except Exception as e:
        logging.error(f"Job processing failed: {e}")
        state["processed_jobs"] = []
        state["job_templates"] = []
        state["diagnostics"] = {
            "job_processing": {
                "status": "error",
                "error": str(e)
            }
        }
    
    return state


if __name__ == "__main__":
    # Test with sample job posting
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Sample job posting
    sample_job = {
        "id": 1,
        "title": "Senior Software Engineer",
        "company": "TechCorp",
        "text": """
        Senior Software Engineer
        TechCorp - San Francisco, CA (Hybrid)
        
        We're looking for a Senior Software Engineer to join our growing team.
        
        Requirements:
        - 5+ years of software development experience
        - Strong proficiency in Python and JavaScript
        - Experience with cloud platforms (AWS preferred)
        - Bachelor's degree in Computer Science or related field
        - Experience with Docker and Kubernetes
        - Strong problem-solving and communication skills
        
        Nice to have:
        - Master's degree
        - AWS certifications
        - Experience with machine learning
        
        Responsibilities:
        - Design and implement scalable backend systems
        - Lead technical initiatives and mentor junior developers
        - Collaborate with product and design teams
        - Participate in code reviews and architecture discussions
        
        We offer:
        - Competitive salary ($150,000 - $200,000)
        - Health, dental, and vision insurance
        - 401(k) with company match
        - Flexible work arrangements (3 days office, 2 days remote)
        - Stock options
        - Professional development budget
        
        Our company values innovation, collaboration, and work-life balance.
        Team size is 15-20 engineers. Fast growth opportunities available.
        
        Visa sponsorship available for qualified candidates.
        """
    }
    
    print("Processing sample job posting...")
    print("-" * 60)
    
    # Test pipeline
    pipeline = JobProcessingPipeline()
    
    try:
        template, diagnostics = pipeline.process_job(sample_job, verbose=True)
        
        print("\n" + "=" * 60)
        print("EXTRACTED INFORMATION")
        print("=" * 60)
        
        print(f"\nJob Title: {template.title}")
        print(f"Company: {template.metadata.get('company', 'N/A')}")
        
        print(f"\nTechnical Skills ({len(template.technical_skills)}):")
        for skill in template.technical_skills:
            print(f"  - {skill.skill_name} ({'Required' if skill.is_mandatory else 'Nice to have'})")
            if skill.evidence:
                print(f"    Evidence: {skill.evidence[0][:50]}...")
        
        print(f"\nExperience Requirements:")
        for exp in template.experience_requirements:
            print(f"  - {exp.description}")
        
        print(f"\nEducation Requirements:")
        for edu in template.education_requirements:
            print(f"  - {edu.degree_level} in {edu.field_of_study or 'any field'} ({'Required' if edu.is_mandatory else 'Preferred'})")
        
        print(f"\nCompensation:")
        print(f"  Salary: ${template.compensation.minimum_salary:,.0f} - ${template.compensation.maximum_salary:,.0f} {template.compensation.currency}")
        print(f"  Benefits: {', '.join(template.compensation.benefits_required[:3])}")
        
        print(f"\nWork Arrangement: {template.location.work_arrangement.value if template.location.work_arrangement else 'Not specified'}")
        print(f"Visa Sponsorship: {'Yes' if template.location.visa_sponsorship_needed else 'No'}")
        
        print(f"\nExtraction Confidence: {template.extraction_confidence:.2f}")
        
        print(f"\nDiagnostics:")
        print(json.dumps(diagnostics, indent=2))
        
    except Exception as e:
        print(f"Error processing job: {e}")
        import traceback
        traceback.print_exc()

"""
Universal Template for Bidirectional Job-Candidate Matching

This module defines the universal template structure that serves as the mapping
between job postings and resumes, enabling bidirectional matching with evidence-based scoring.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class WorkArrangement(Enum):
    """Work arrangement preferences"""
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    FLEXIBLE = "flexible"


class EmploymentType(Enum):
    """Employment type preferences"""
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    FREELANCE = "freelance"
    INTERNSHIP = "internship"


@dataclass
class LocationRequirement:
    """Location preferences and requirements"""
    preferred_locations: List[str] = field(default_factory=list)
    work_arrangement: Optional[WorkArrangement] = None
    relocation_willing: bool = False
    visa_sponsorship_needed: bool = False
    work_authorization: Optional[str] = None


@dataclass
class CompensationRequirement:
    """Compensation expectations and offers"""
    minimum_salary: Optional[float] = None
    maximum_salary: Optional[float] = None
    currency: str = "USD"
    benefits_required: List[str] = field(default_factory=list)  # health, dental, 401k, etc.
    equity_expectation: Optional[str] = None
    bonus_structure: Optional[str] = None


@dataclass
class SkillRequirement:
    """Skill requirement with proficiency and evidence"""
    skill_name: str
    required_proficiency: str  # beginner, intermediate, advanced, expert
    years_required: Optional[float] = None
    is_mandatory: bool = True
    evidence: List[str] = field(default_factory=list)  # Evidence from source document
    confidence: float = 0.0  # 0-1 confidence score


@dataclass
class ExperienceRequirement:
    """Experience requirement with evidence"""
    experience_type: str  # e.g., "management", "technical", "industry"
    years_required: float
    description: str
    is_mandatory: bool = True
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class EducationRequirement:
    """Education requirement with evidence"""
    degree_level: str  # e.g., "Bachelor's", "Master's", "PhD"
    field_of_study: Optional[str] = None
    is_mandatory: bool = True
    alternatives_accepted: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class CultureFit:
    """Cultural and work environment preferences"""
    company_values: List[str] = field(default_factory=list)
    work_life_balance: Optional[str] = None
    team_size_preference: Optional[str] = None
    management_style: Optional[str] = None
    career_growth_importance: Optional[str] = None
    industry_preferences: List[str] = field(default_factory=list)
    company_size_preference: Optional[str] = None  # startup, mid-size, enterprise


@dataclass
class VocationTemplate:
    """
    Universal template for bidirectional job-candidate matching
    
    This template can be filled from either a job posting or a resume,
    with evidence and confidence scores for each extracted field.
    """
    
    # Core Identification
    source_type: str  # "job_posting" or "resume"
    source_id: str
    title: str  # Job title or candidate's current/desired title
    
    # Skills & Qualifications
    technical_skills: List[SkillRequirement] = field(default_factory=list)
    soft_skills: List[SkillRequirement] = field(default_factory=list)
    certifications: List[str] = field(default_factory=list)
    tools_technologies: List[SkillRequirement] = field(default_factory=list)
    
    # Experience
    experience_requirements: List[ExperienceRequirement] = field(default_factory=list)
    total_years_experience: Optional[float] = None
    management_experience: Optional[float] = None
    
    # Education
    education_requirements: List[EducationRequirement] = field(default_factory=list)
    
    # Location & Work Arrangement
    location: LocationRequirement = field(default_factory=LocationRequirement)
    
    # Compensation
    compensation: CompensationRequirement = field(default_factory=CompensationRequirement)
    
    # Employment Details
    employment_type: Optional[EmploymentType] = None
    start_date: Optional[str] = None
    
    # Culture & Fit
    culture_fit: CultureFit = field(default_factory=CultureFit)
    
    # Responsibilities (for jobs) / Achievements (for candidates)
    key_responsibilities: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    
    # Free-form Summary
    summary: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_confidence: float = 0.0  # Overall confidence in extraction
    processed_at: Optional[str] = None
    
    def to_matching_vector(self) -> Dict[str, Any]:
        """Convert template to a vector representation for matching"""
        return {
            "skills": [s.skill_name for s in self.technical_skills + self.soft_skills],
            "experience_years": self.total_years_experience,
            "education_level": [e.degree_level for e in self.education_requirements],
            "location_flexibility": self.location.work_arrangement,
            "salary_range": (self.compensation.minimum_salary, self.compensation.maximum_salary),
            "must_have_skills": [s.skill_name for s in self.technical_skills if s.is_mandatory],
            "nice_to_have_skills": [s.skill_name for s in self.technical_skills if not s.is_mandatory],
        }


@dataclass
class BidirectionalMatch:
    """Result of bidirectional matching between job and candidate"""
    
    job_template: VocationTemplate
    candidate_template: VocationTemplate
    
    # Job -> Candidate fit (Is this candidate right for the job?)
    job_fit_score: float = 0.0  # 0-1 score
    job_fit_reasons: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    exceeds_requirements: List[str] = field(default_factory=list)
    
    # Candidate -> Job fit (Is this job right for the candidate?)
    candidate_fit_score: float = 0.0  # 0-1 score
    candidate_fit_reasons: List[str] = field(default_factory=list)
    unmet_preferences: List[str] = field(default_factory=list)
    preference_matches: List[str] = field(default_factory=list)
    
    # Overall mutual fit
    mutual_fit_score: float = 0.0  # Geometric mean of both scores
    recommendation: str = ""  # "strong_match", "good_match", "partial_match", "poor_match"
    action_items: List[str] = field(default_factory=list)  # Suggested next steps
    
    # Detailed scoring breakdown
    skill_alignment: float = 0.0
    experience_alignment: float = 0.0
    education_alignment: float = 0.0
    compensation_alignment: float = 0.0
    location_alignment: float = 0.0
    culture_alignment: float = 0.0
    
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JobSeekerPreferences:
    """Job seeker preferences for matching"""
    
    # Job preferences
    desired_titles: List[str] = field(default_factory=list)
    avoid_titles: List[str] = field(default_factory=list)
    
    # Company preferences
    preferred_companies: List[str] = field(default_factory=list)
    avoid_companies: List[str] = field(default_factory=list)
    preferred_industries: List[str] = field(default_factory=list)
    avoid_industries: List[str] = field(default_factory=list)
    
    # Work preferences
    minimum_acceptable_salary: Optional[float] = None
    preferred_work_arrangement: Optional[WorkArrangement] = None
    acceptable_work_arrangements: List[WorkArrangement] = field(default_factory=list)
    
    # Importance weights (0-1)
    salary_importance: float = 0.7
    location_importance: float = 0.6
    work_life_balance_importance: float = 0.8
    career_growth_importance: float = 0.9
    company_culture_importance: float = 0.7
    
    # Deal breakers
    deal_breakers: List[str] = field(default_factory=list)
    
    # Additional preferences
    preferred_team_size: Optional[str] = None
    preferred_company_size: Optional[str] = None
    travel_willingness: Optional[str] = None  # none, occasional, frequent
    on_call_willingness: bool = False
    
    metadata: Dict[str, Any] = field(default_factory=dict)

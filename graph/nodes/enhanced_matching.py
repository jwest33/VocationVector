"""
Enhanced Job Matching System - Feature-by-feature bidirectional matching with gap analysis
Uses section-level embeddings and detailed feature comparison
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path

from sklearn.metrics.pairwise import cosine_similarity
from graph.embeddings import JobEmbeddings
from graph.database import graphDB
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


@dataclass
class DetailedMatch:
    """Comprehensive match result with feature-level scores"""
    # Identifiers
    job_id: str
    resume_id: str
    job_title: str
    company: str
    candidate_name: str
    
    # Overall scores
    overall_score: float
    job_fit_score: float  # How well resume fits job
    candidate_fit_score: float  # How well job fits candidate preferences
    
    # Feature-level scores
    title_alignment: float
    skills_match: float
    experience_match: float
    education_match: float
    location_match: float
    salary_alignment: float
    
    # Section-level semantic scores
    summary_to_description: float
    experience_to_requirements: float
    skills_to_skills: float
    
    # Detailed analysis
    matched_skills: List[Dict[str, Any]]
    missing_skills: List[str]
    exceeded_skills: List[str]  # Skills candidate has that job doesn't require
    
    matched_requirements: List[str]
    missing_requirements: List[str]
    
    education_gaps: List[str]
    experience_gaps: Dict[str, Any]
    
    # Preferences alignment
    salary_match: Dict[str, Any]
    location_preference_met: bool
    remote_preference_met: bool
    
    # LLM assessment
    llm_score: float
    llm_reasoning: str
    llm_recommendations: List[str]
    
    # Metadata
    match_timestamp: str
    confidence_score: float


class EnhancedMatcher:
    """Advanced matching system with bidirectional feature comparison"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Ensure config is always a dictionary
        if config is None:
            self.config = {}
        elif isinstance(config, dict):
            self.config = config
        elif hasattr(config, '__dict__'):
            self.config = config.__dict__
        else:
            logger.warning(f"Unexpected config type: {type(config)}, using empty dict")
            self.config = {}
            
        self.embedder = JobEmbeddings()
        self.db = graphDB()
        
        # Load user preferences from app
        self.user_preferences = self._load_user_preferences()
        
        # Initialize LLM client - now config is guaranteed to be a dict
        import os
        base_url = self.config.get('llm_base_url', os.getenv("OPENAI_BASE_URL", 'http://localhost:8000/v1'))
        model = self.config.get('llm_model', os.getenv("LLM_MODEL", 'qwen3-4b-instruct-2507-f16'))
        
        logger.info(f"Initializing LLM client with base_url: {base_url}, model: {model}")
        
        # Test connection to LLM server
        try:
            import requests
            response = requests.get(f"{base_url}/models", timeout=2)
            if response.status_code == 200:
                logger.info(f"LLM server is accessible at {base_url}")
            else:
                logger.warning(f"LLM server returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not verify LLM server connection: {e}")
        
        self.llm_client = OpenAI(
            base_url=base_url,
            api_key="dummy-key"
        )
        self.llm_model = model
        
        # Scoring weights
        self.weights = {
            'title': 0.15,
            'skills': 0.25,
            'experience': 0.20,
            'education': 0.10,
            'semantic': 0.20,
            'llm': 0.10
        }
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user preferences from the app preferences file"""
        try:
            prefs_file = Path('data/preferences.json')
            if prefs_file.exists():
                with open(prefs_file, 'r') as f:
                    prefs = json.load(f)
                    logger.info(f"Loaded user preferences: {prefs}")
                    print(f"Loaded user preferences: {prefs}")
                    return prefs
        except Exception as e:
            logger.warning(f"Could not load user preferences: {e}")
        logger.info("No user preferences found, using defaults")
        print("No user preferences found, using defaults")
        return {}
    
    def match_all(self, resume_ids: Optional[List[str]] = None, 
                  job_ids: Optional[List[str]] = None,
                  top_k: int = 1000) -> List[DetailedMatch]:
        """Perform comprehensive matching between resumes and jobs"""
        
        print("Loading resumes from database...")
        # Get data from database
        resumes = self._get_resumes(resume_ids)
        print(f"Loaded {len(resumes)} resumes")
        
        print("Loading jobs from database...")
        jobs = self._get_jobs(job_ids)
        print(f"Loaded {len(jobs)} jobs")
        
        if not resumes or not jobs:
            logger.warning("No resumes or jobs to match")
            return []
        
        print(f"Calculating compatibility scores for {len(resumes)} resumes × {len(jobs)} jobs...")
        all_matches = []
        
        for i, resume in enumerate(resumes, 1):
            if i % 5 == 0 or i == 1:
                print(f"Processing resume {i}/{len(resumes)}...")
            resume_matches = []
            
            for job in jobs:
                match = self._match_single(resume, job)
                # Use min_match_score from config, default to 0.3
                min_score = self.config.get('min_match_score', self.config.get('min_score', 0.3))
                if match and match.overall_score >= min_score:
                    resume_matches.append(match)
            
            # Sort by score and take top K
            resume_matches.sort(key=lambda x: x.overall_score, reverse=True)
            all_matches.extend(resume_matches[:top_k])
        
        return all_matches
    
    def _match_single(self, resume: Dict[str, Any], job: Dict[str, Any]) -> DetailedMatch:
        """Perform detailed matching between a single resume and job"""
        
        # Extract structured data
        resume_template = self._parse_json_field(resume.get('vocation_template'))
        job_template = self._parse_json_field(job.get('vocation_template'))
        
        # Parse matching_template properly
        matching_template = self._parse_json_field(resume.get('matching_template'), {})
        
        # 1. Title Alignment
        title_score = self._match_title(
            matching_template,
            job.get('title', '')
        )
        
        # 2. Skills Matching
        skills_analysis = self._match_skills(resume, job)
        
        # 3. Experience Matching
        experience_analysis = self._match_experience(resume, job)
        
        # 4. Education Matching
        education_analysis = self._match_education(resume, job)
        
        # 5. Location & Remote Preferences
        location_analysis = self._match_location(resume, job)
        
        # 6. Salary Alignment
        salary_analysis = self._match_salary(resume, job)
        
        # 7. Section-level Semantic Matching
        semantic_scores = self._compute_semantic_scores(resume, job)
        
        # 8. LLM Assessment
        llm_analysis = self._get_llm_assessment(resume, job, {
            'skills': skills_analysis,
            'experience': experience_analysis,
            'education': education_analysis
        })
        
        # Calculate bidirectional scores
        job_fit_score = self._calculate_job_fit(
            title_score, skills_analysis, experience_analysis, 
            education_analysis, semantic_scores
        )
        
        candidate_fit_score = self._calculate_candidate_fit(
            location_analysis, salary_analysis, job_template
        )
        
        # Overall score combines both directions
        overall_score = (job_fit_score * 0.7 + candidate_fit_score * 0.3)
        
        return DetailedMatch(
            job_id=job.get('job_id', ''),
            resume_id=resume.get('resume_id', ''),
            job_title=job.get('title', 'Unknown Position'),
            company=job.get('company', 'Unknown Company'),
            candidate_name=resume.get('name', 'Unknown Candidate'),
            
            overall_score=overall_score,
            job_fit_score=job_fit_score,
            candidate_fit_score=candidate_fit_score,
            
            title_alignment=title_score,
            skills_match=skills_analysis['score'],
            experience_match=experience_analysis['score'],
            education_match=education_analysis['score'],
            location_match=location_analysis['score'],
            salary_alignment=salary_analysis['score'],
            
            summary_to_description=semantic_scores['summary_to_description'],
            experience_to_requirements=semantic_scores['experience_to_requirements'],
            skills_to_skills=semantic_scores['skills_to_skills'],
            
            matched_skills=skills_analysis['matched'],
            missing_skills=skills_analysis['missing'],
            exceeded_skills=skills_analysis['exceeded'],
            
            matched_requirements=experience_analysis['matched'],
            missing_requirements=experience_analysis['missing'],
            
            education_gaps=education_analysis['gaps'],
            experience_gaps=experience_analysis['gaps'],
            
            salary_match=salary_analysis,
            location_preference_met=location_analysis['preference_met'],
            remote_preference_met=location_analysis['remote_met'],
            
            llm_score=llm_analysis['score'],
            llm_reasoning=llm_analysis['reasoning'],
            llm_recommendations=llm_analysis['recommendations'],
            
            match_timestamp=datetime.now().isoformat(),
            confidence_score=self._calculate_confidence(semantic_scores, skills_analysis)
        )
    
    def _match_skills(self, resume: Dict, job: Dict) -> Dict[str, Any]:
        """Detailed skills comparison with fuzzy matching"""
        
        # Parse skills from both sides
        resume_skills = self._parse_json_field(resume.get('skills', '{}'))
        job_skills = self._parse_json_field(job.get('skills', '[]'))
        
        # Extract skill names
        resume_skill_list = []
        if isinstance(resume_skills, dict):
            for category in ['technical', 'languages', 'tools']:
                resume_skill_list.extend(resume_skills.get(category, []))
        elif isinstance(resume_skills, list):
            resume_skill_list = list(resume_skills)
        
        job_skill_list = list(job_skills) if isinstance(job_skills, list) else []
        
        # Fuzzy matching for skills
        matched = []
        matched_job_indices = set()
        matched_resume_indices = set()
        
        # First pass: exact matches
        for i, job_skill in enumerate(job_skill_list):
            for j, resume_skill in enumerate(resume_skill_list):
                if i not in matched_job_indices and j not in matched_resume_indices:
                    if self._skills_match(job_skill, resume_skill):
                        matched.append({'skill': job_skill, 'matched_with': resume_skill})
                        matched_job_indices.add(i)
                        matched_resume_indices.add(j)
        
        # Identify missing and exceeded skills
        missing = [skill for i, skill in enumerate(job_skill_list) if i not in matched_job_indices]
        exceeded = [skill for i, skill in enumerate(resume_skill_list) if i not in matched_resume_indices]
        
        # Score calculation
        if job_skill_list:
            score = len(matched) / len(job_skill_list)
        else:
            score = 0.5  # No specific skills required
        
        # Boost score if candidate has extra relevant skills
        if exceeded:
            score = min(1.0, score + len(exceeded) * 0.02)
        
        return {
            'score': score,
            'matched': matched,
            'missing': missing,
            'exceeded': exceeded,
            'match_rate': f"{len(matched)}/{len(job_skill_list)}" if job_skill_list else "N/A"
        }
    
    def _skills_match(self, skill1: str, skill2: str) -> bool:
        """Check if two skills match (with fuzzy matching)"""
        if not skill1 or not skill2:
            return False
            
        # Normalize for comparison
        s1 = str(skill1).lower().strip()
        s2 = str(skill2).lower().strip()
        
        # Exact match
        if s1 == s2:
            return True
        
        # Remove common suffixes/prefixes and parenthetical info
        s1_base = s1.split('(')[0].strip()
        s2_base = s2.split('(')[0].strip()
        
        # Check if base matches
        if s1_base == s2_base:
            return True
        
        # Check if one contains the other (e.g., "Excel" in "Excel (Advanced)")
        if s1_base in s2 or s2_base in s1:
            return True
        
        # Common variations
        variations = {
            'js': 'javascript',
            'ts': 'typescript', 
            'py': 'python',
            'c#': 'csharp',
            'c++': 'cpp',
            'node': 'nodejs',
            'node.js': 'nodejs',
            'react.js': 'react',
            'vue.js': 'vue',
            'angular.js': 'angular'
        }
        
        # Check with variations
        s1_normalized = variations.get(s1_base, s1_base)
        s2_normalized = variations.get(s2_base, s2_base)
        
        if s1_normalized == s2_normalized:
            return True
            
        return False
    
    def _match_experience(self, resume: Dict, job: Dict) -> Dict[str, Any]:
        """Compare experience requirements"""
        
        resume_years = resume.get('years_experience', 0) or 0
        job_template = self._parse_json_field(job.get('vocation_template', '{}'))
        # Try multiple fields for required years
        required_years = (job_template.get('years_experience_required') or 
                         job_template.get('total_years_experience') or 0)
        
        # Parse requirements
        job_requirements = self._parse_json_field(job.get('requirements', '[]'))
        resume_experience = self._parse_json_field(resume.get('experience', '[]'))
        
        # Years comparison - Don't penalize for having more experience
        years_diff = resume_years - required_years
        if years_diff >= 0:
            # Has enough or more experience
            if years_diff <= 5:
                years_score = 1.0  # Perfect: meets requirement with 0-5 years extra
            elif years_diff <= 10:
                years_score = 0.9  # Slightly overqualified: 5-10 years extra
            else:
                years_score = 0.8  # May be overqualified: 10+ years extra
        elif years_diff >= -2:
            years_score = 0.7  # Close enough: within 2 years
        else:
            years_score = max(0.3, 1.0 - abs(years_diff) * 0.1)  # Under-qualified
        
        # Requirements matching - filter out years requirements and education requirements
        matched_reqs = []
        missing_reqs = []
        
        if job_requirements and resume_experience:
            for req in job_requirements:
                # Skip None or empty requirements
                if not req:
                    continue
                
                req_str = str(req).lower()
                
                # Skip education requirements (handled separately in _match_education)
                education_keywords = ['bachelor', 'master', 'phd', 'doctorate', 'associate', 
                                    'degree', 'diploma', 'education', 'university', 'college']
                if any(keyword in req_str for keyword in education_keywords):
                    continue  # Skip education requirements in experience matching
                
                # Check if this is a years requirement (skip it, handled above)
                if 'years experience' in req_str or 'years of experience' in req_str:
                    # Years requirement - check if met
                    if years_diff >= 0:
                        matched_reqs.append(req)
                    else:
                        missing_reqs.append(req)
                # Otherwise check if requirement is met by experience
                elif self._requirement_met_by_experience(req, resume_experience):
                    matched_reqs.append(req)
                else:
                    missing_reqs.append(req)
        
        req_score = len(matched_reqs) / len(job_requirements) if job_requirements else 0.5
        
        return {
            'score': (years_score * 0.6 + req_score * 0.4),
            'matched': matched_reqs,
            'missing': missing_reqs,
            'gaps': {
                'years_difference': years_diff,
                'years_score': years_score,
                'requirements_met': f"{len(matched_reqs)}/{len(job_requirements)}" if job_requirements else "N/A"
            }
        }
    
    def _match_education(self, resume: Dict, job: Dict) -> Dict[str, Any]:
        """Compare education requirements"""
        
        resume_education = self._parse_json_field(resume.get('education', '[]'))
        job_template = self._parse_json_field(job.get('vocation_template', '{}'))
        job_education = job_template.get('education_requirements', [])
        
        if not job_education:
            return {'score': 1.0, 'gaps': [], 'met': True}
        
        # Debug logging
        logger.debug(f"Education matching for {resume.get('name', 'Unknown')}")
        logger.debug(f"  Resume education: {resume_education}")
        logger.debug(f"  Job requirements: {job_education}")
        
        gaps = []
        met_requirements = []
        
        for req in job_education:
            # Skip None or invalid requirements
            if not req or not isinstance(req, dict):
                continue
            
            is_met = self._education_requirement_met(req, resume_education)
            logger.debug(f"  Requirement {req} met: {is_met}")
            
            if is_met:
                met_requirements.append(req)
            else:
                gaps.append(f"{req.get('degree', 'Degree')} in {req.get('field', 'Any field')}")
        
        score = len(met_requirements) / len(job_education) if job_education else 1.0
        
        logger.debug(f"  Education score: {score}, gaps: {gaps}")
        
        return {
            'score': score,
            'gaps': gaps,
            'met': len(gaps) == 0,
            'requirements_met': f"{len(met_requirements)}/{len(job_education)}"
        }
    
    def _match_location(self, resume: Dict, job: Dict) -> Dict[str, Any]:
        """Match location and remote preferences"""
        
        # Use app preferences if available, otherwise check resume
        prefs = self.user_preferences if self.user_preferences else {}
        job_location = job.get('location', '') or ''
        job_remote = job.get('remote_policy', '') or ''
        
        # Debug output
        logger.debug(f"Location matching - Job: {job.get('title')} at {job.get('company')}")
        logger.debug(f"  Job location: {job_location}")
        logger.debug(f"  Job remote policy: {job_remote}")
        logger.debug(f"  User preferences: {prefs.get('locations')}")
        
        # Also check if job location itself says remote
        job_is_remote = ('remote' in job_remote.lower() if job_remote else False) or ('remote' in job_location.lower())
        
        # Check remote preference from app
        prefers_remote = False
        if prefs.get('locations'):
            # Check if any preferred location includes "remote"
            prefers_remote = any('remote' in loc.lower() for loc in prefs['locations'] if loc)
        
        logger.debug(f"  Job is remote: {job_is_remote}")
        logger.debug(f"  User prefers remote: {prefers_remote}")
        
        # If user prefers remote, job must be remote. If not, any job is OK
        remote_match = job_is_remote if prefers_remote else True
        
        # Location matching
        location_match = True
        if prefs.get('locations'):
            preferred_locations = prefs['locations']
            # If job is remote and user wants remote, that's a match
            if job_is_remote and prefers_remote:
                location_match = True
            # Otherwise check if job location matches preferences
            elif job_location and preferred_locations:
                location_match = any(
                    loc.lower() in job_location.lower() or job_location.lower() in loc.lower()
                    for loc in preferred_locations 
                    if loc and isinstance(loc, str) and loc.lower() != 'remote'
                )
        
        score = 1.0 if (remote_match and location_match) else 0.5
        
        return {
            'score': score,
            'preference_met': location_match,
            'remote_met': remote_match,
            'job_location': job_location,
            'job_remote_policy': job_remote
        }
    
    def _match_salary(self, resume: Dict, job: Dict) -> Dict[str, Any]:
        """Compare salary expectations"""
        
        # Use app preferences for salary if available
        prefs = self.user_preferences if self.user_preferences else {}
        
        # Try to get salary from preferences first, then resume
        if prefs.get('min_salary') or prefs.get('max_salary'):
            expected_min = prefs.get('min_salary', 0)
            expected_max = prefs.get('max_salary', float('inf'))
        else:
            resume_salary = self._parse_json_field(resume.get('salary_expectations', '{}'))
            if resume_salary:
                expected_min = resume_salary.get('minimum', 0)
                expected_max = resume_salary.get('maximum', float('inf'))
            else:
                expected_min = 0
                expected_max = float('inf')
        
        job_min = job.get('salary_min')
        job_max = job.get('salary_max')
        
        # Check if job_min/job_max are nan or None
        import math
        if job_min is not None and math.isnan(job_min):
            job_min = None
        if job_max is not None and math.isnan(job_max):
            job_max = None
        
        # If job has no salary info, it's not aligned (user wants transparency)
        if not job_min and not job_max:
            return {
                'score': 0.0,
                'alignment': 'not_specified',
                'job_range': 'Not specified',
                'expected_range': f"${expected_min:,.0f}-${expected_max:,.0f}" if expected_min and expected_min != float('inf') else 'Flexible'
            }
        
        # If user has no expectations, be neutral
        if expected_min == 0 and expected_max == float('inf'):
            return {
                'score': 0.5,
                'alignment': 'no_preference',
                'job_range': f"${job_min:,.0f}-${job_max:,.0f}" if job_min and job_max else 'Partial data',
                'expected_range': 'Flexible'
            }
        
        # Check overlap
        if job_max and expected_min:
            if job_max < expected_min:
                score = 0.3  # Job pays less than expected
                alignment = 'below_expectations'
            elif job_min and job_min > expected_max:
                score = 0.3  # Job pays more than expected (might be overqualified)
                alignment = 'above_expectations'
            else:
                score = 1.0  # Good alignment
                alignment = 'aligned'
        else:
            score = 0.5
            alignment = 'partial_data'
        
        # Format ranges safely
        if job_min and job_max:
            job_range = f"${job_min:,.0f}-${job_max:,.0f}"
        elif job_min:
            job_range = f"${job_min:,.0f}+"
        elif job_max:
            job_range = f"Up to ${job_max:,.0f}"
        else:
            job_range = 'Not specified'
        
        if expected_min and expected_max != float('inf'):
            expected_range = f"${expected_min:,.0f}-${expected_max:,.0f}"
        elif expected_min:
            expected_range = f"${expected_min:,.0f}+"
        else:
            expected_range = 'Flexible'
        
        return {
            'score': score,
            'alignment': alignment,
            'job_range': job_range,
            'expected_range': expected_range
        }
    
    def _compute_semantic_scores(self, resume: Dict, job: Dict) -> Dict[str, float]:
        """Calculate section-level semantic similarities"""
        
        scores = {}
        
        # Helper function to check if embedding exists and is valid
        def has_embedding(data: Dict, key: str) -> bool:
            embedding = data.get(key)
            if embedding is None:
                return False
            # Check if it's a non-empty list/array
            try:
                return len(embedding) > 0
            except:
                return False
        
        # Summary to Description
        if has_embedding(resume, 'embedding_summary') and has_embedding(job, 'embedding_description'):
            scores['summary_to_description'] = self._cosine_similarity(
                resume['embedding_summary'],
                job['embedding_description']
            )
        else:
            scores['summary_to_description'] = 0.0
        
        # Experience to Requirements
        if has_embedding(resume, 'embedding_experience') and has_embedding(job, 'embedding_requirements'):
            scores['experience_to_requirements'] = self._cosine_similarity(
                resume['embedding_experience'],
                job['embedding_requirements']
            )
        else:
            scores['experience_to_requirements'] = 0.0
        
        # Skills to Skills
        if has_embedding(resume, 'embedding_skills') and has_embedding(job, 'embedding_skills'):
            scores['skills_to_skills'] = self._cosine_similarity(
                resume['embedding_skills'],
                job['embedding_skills']
            )
        else:
            scores['skills_to_skills'] = 0.0
        
        # Overall semantic similarity
        if has_embedding(resume, 'embedding_full') and has_embedding(job, 'embedding_full'):
            scores['overall'] = self._cosine_similarity(
                resume['embedding_full'],
                job['embedding_full']
            )
        else:
            scores['overall'] = 0.0
        
        return scores
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _get_llm_assessment(self, resume: Dict, job: Dict, analysis: Dict) -> Dict[str, Any]:
        """Get LLM assessment of the match with gap analysis"""
        
        logger.info(f"Starting LLM assessment for {resume.get('name', 'Unknown')} -> {job.get('title', 'Unknown')}")
        logger.info(f"LLM client configured with: base_url={self.llm_client.base_url}, model={self.llm_model}")
        
        # Parse job data safely
        job_title = job.get('title', 'Unknown Position')
        job_company = job.get('company', 'Unknown Company')
        
        # Parse skills from job
        job_skills = self._parse_json_field(job.get('skills', []))
        if isinstance(job_skills, list):
            job_skills_str = ', '.join(str(s) for s in job_skills[:10])  # Limit to 10 skills
        else:
            job_skills_str = 'Not specified'
        
        # Parse universal template for experience and education requirements
        job_template = self._parse_json_field(job.get('vocation_template', {}))
        years_required = 'Not specified'
        education_required = 'Not specified'
        if isinstance(job_template, dict):
            years_required = job_template.get('total_years_experience', 'Not specified')
            # Parse education requirements
            edu_reqs = job_template.get('education_requirements', [])
            if edu_reqs and isinstance(edu_reqs, list):
                edu_req_list = []
                for req in edu_reqs:
                    if isinstance(req, dict):
                        degree = req.get('degree', '')
                        field = req.get('field', '')
                        if degree:
                            edu_req_list.append(f"{degree} in {field}" if field else degree)
                education_required = '; '.join(edu_req_list) if edu_req_list else 'Not specified'
        
        # Parse resume data safely
        resume_name = resume.get('name', 'Unknown')
        resume_years = resume.get('years_experience', 0)
        
        # Parse skills from resume
        resume_skills = self._parse_json_field(resume.get('skills', []))
        if isinstance(resume_skills, list):
            resume_skills_str = ', '.join(str(s) for s in resume_skills[:10])  # Limit to 10 skills
        elif isinstance(resume_skills, dict):
            all_skills = []
            for category in ['technical', 'languages', 'tools']:
                all_skills.extend(resume_skills.get(category, []))
            resume_skills_str = ', '.join(str(s) for s in all_skills[:10])
        else:
            resume_skills_str = 'Not specified'
        
        # Parse education from resume
        resume_education = self._parse_json_field(resume.get('education', []))
        education_str = 'Not specified'
        if isinstance(resume_education, list) and resume_education:
            edu_list = []
            for edu in resume_education[:3]:  # Limit to 3 degrees
                if isinstance(edu, dict):
                    degree = edu.get('degree', '')
                    institution = edu.get('institution', '')
                    if degree:
                        edu_list.append(f"{degree} from {institution}" if institution else degree)
            education_str = '; '.join(edu_list) if edu_list else 'Not specified'
        
        # Format analysis data safely
        matched_skills = analysis.get('skills', {}).get('matched', [])
        if isinstance(matched_skills, list):
            matched_skills_str = ', '.join(str(s) for s in matched_skills[:5]) if matched_skills else 'None'
        else:
            matched_skills_str = 'None'
            
        missing_skills = analysis.get('skills', {}).get('missing', [])
        if isinstance(missing_skills, list):
            missing_skills_str = ', '.join(str(s) for s in missing_skills[:5]) if missing_skills else 'None'
        else:
            missing_skills_str = 'None'
        
        experience_gaps = analysis.get('experience', {}).get('gaps', {})
        if isinstance(experience_gaps, dict):
            gaps_str = str(experience_gaps).replace('{', '').replace('}', '')[:100]  # Limit length
        else:
            gaps_str = 'None identified'
        
        # Log what we're sending to the LLM for debugging
        logger.info(f"Education being sent to LLM - Job requires: {education_required}")
        logger.info(f"Education being sent to LLM - Resume has: {education_str}")
        
        prompt = f"""Analyze this job match from the CANDIDATE'S perspective:

JOB: {job_title} at {job_company}
Skills Required: {job_skills_str}
Experience Required: {years_required} years
Education Required: {education_required}

YOUR PROFILE:
Skills: {resume_skills_str}
Experience: {resume_years} years
Education: {education_str}

MATCH ANALYSIS:
- Your Matching Skills: {matched_skills_str}
- Skills to Develop: {missing_skills_str}
- Experience Comparison: You have {resume_years} years, job requires {years_required} years
- Education: Please evaluate if "{education_str}" meets the requirement for "{education_required}". Be specific about whether Bachelor of Business Administration satisfies a requirement for Bachelor's in Business Administration or related field.

As a career advisor to the candidate, provide:
1. Match score (0-1) 
2. Your key strengths for this role (2-3 points)
3. Gaps you should address (2-3 points) - ONLY list real gaps where you don't meet requirements
4. Actions YOU should take to improve your candidacy

IMPORTANT: Evaluate education match carefully:
- If candidate has "Bachelor of Business Administration" and job requires "Bachelor's in Business Administration", that's a MATCH
- Only mark education as a gap if degree level or field truly doesn't match

Return JSON only:
{{
  "score": 0.0-1.0,
  "reasoning": "Brief explanation including whether education requirement is met",
  "strengths": ["Key strength for this role", "Another relevant strength"],
  "gaps": ["Only list actual gaps - not education if you have the required degree"],
  "recommendations": ["Action you should take", "Another step to improve your candidacy"]
}}"""

        try:
            logger.info(f"Sending LLM request for {resume_name} -> {job_title}")
            logger.info(f"LLM Model: {self.llm_model}, Prompt length: {len(prompt)} characters")
            
            # Try to make the LLM call
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": "You are an expert recruiter. Analyze the match and return ONLY valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=500,
                    timeout=30
                )
                logger.info(f"LLM response received for {resume_name} -> {job_title}")
            except Exception as api_error:
                logger.error(f"LLM API call failed: {api_error}")
                logger.error(f"API error type: {type(api_error).__name__}")
                raise  # Re-raise to trigger retry
            
            # Get response content
            content = response.choices[0].message.content
            
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group()
            
            result = json.loads(content)
            
            return {
                'score': float(result.get('score', 0.5)),
                'reasoning': str(result.get('reasoning', 'Assessment generated')),
                'recommendations': result.get('recommendations', [])
            }
            
        except json.JSONDecodeError as e:
            logger.error(f"LLM returned invalid JSON: {e}")
            logger.error(f"LLM response was: {content[:500] if 'content' in locals() else 'No content'}")
            return {
                'score': 0.5,
                'reasoning': 'Unable to parse LLM assessment',
                'recommendations': []
            }
        except Exception as e:
            logger.error(f"LLM assessment failed: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {
                'score': 0.5,
                'reasoning': f'LLM assessment unavailable: {str(e)[:100]}',
                'recommendations': []
            }
    
    def _calculate_job_fit(self, title_score: float, skills: Dict, 
                          experience: Dict, education: Dict, 
                          semantic: Dict) -> float:
        """Calculate how well candidate fits the job"""
        
        return (
            title_score * 0.15 +
            skills['score'] * 0.25 +
            experience['score'] * 0.20 +
            education['score'] * 0.10 +
            semantic.get('overall', 0) * 0.30
        )
    
    def _calculate_candidate_fit(self, location: Dict, salary: Dict, 
                                 job_template: Dict) -> float:
        """Calculate how well job fits candidate preferences"""
        
        return (
            location['score'] * 0.4 +
            salary['score'] * 0.4 +
            self._calculate_growth_fit(job_template) * 0.2
        )
    
    def _calculate_growth_fit(self, job_template: Dict) -> float:
        """Assess growth opportunities"""
        
        if job_template.get('growth_opportunities'):
            return 0.8
        return 0.5
    
    def _calculate_confidence(self, semantic: Dict, skills: Dict) -> float:
        """Calculate confidence in the match"""
        
        # Higher confidence when multiple signals agree
        signals = [
            semantic.get('overall', 0) > 0.5,
            skills['score'] > 0.5,
            len(skills['matched']) > 3
        ]
        
        return sum(signals) / len(signals)
    
    def _match_title(self, resume_template: Dict, job_title: str) -> float:
        """Match job title with user's preferred titles or resume title using semantic embeddings"""
        
        if not job_title:
            return 0.5
        
        # Ensure job_title is a string
        if not isinstance(job_title, str):
            return 0.5
        
        job_title_lower = job_title.lower()
        job_keywords = self._extract_title_keywords(job_title_lower)
        
        # First check user preferences for preferred titles
        if self.user_preferences and 'preferred_titles' in self.user_preferences:
            preferred_titles = self.user_preferences['preferred_titles']
            if preferred_titles and isinstance(preferred_titles, list):
                best_score = 0.0
                titles_to_compare = []
                
                for preferred in preferred_titles:
                    if preferred and isinstance(preferred, str) and preferred.strip():
                        preferred_lower = preferred.lower().strip()
                        
                        # Direct substring match - still keep this for exact matches
                        if preferred_lower in job_title_lower or job_title_lower in preferred_lower:
                            return 1.0
                        
                        titles_to_compare.append(preferred)
                
                # Use semantic embeddings for non-exact matches
                if titles_to_compare:
                    try:
                        # Encode job title and preferred titles
                        job_embedding = self.embedder.encode(job_title)
                        preferred_embeddings = self.embedder.encode(titles_to_compare)
                        
                        # Calculate cosine similarities
                        similarities = cosine_similarity(
                            job_embedding.reshape(1, -1),
                            preferred_embeddings
                        )[0]
                        
                        # Get best semantic match
                        best_semantic = float(max(similarities))
                        
                        # Combine with keyword matching for hybrid score
                        for i, preferred in enumerate(titles_to_compare):
                            preferred_keywords = self._extract_title_keywords(preferred.lower())
                            keyword_score = 0
                            
                            if preferred_keywords and job_keywords:
                                overlap = len(preferred_keywords.intersection(job_keywords))
                                max_possible = min(len(preferred_keywords), len(job_keywords))
                                if max_possible > 0:
                                    keyword_score = overlap / max_possible
                                    if self._has_role_match(preferred_keywords, job_keywords):
                                        keyword_score = min(1.0, keyword_score + 0.2)
                            
                            # Weighted combination: 70% semantic, 30% keyword
                            combined_score = (similarities[i] * 0.7) + (keyword_score * 0.3)
                            best_score = max(best_score, combined_score)
                        
                        # Also consider pure semantic score if it's very high
                        best_score = max(best_score, best_semantic)
                        
                    except Exception as e:
                        logger.warning(f"Error in semantic title matching: {e}")
                        # Fall back to keyword matching
                        for preferred in titles_to_compare:
                            preferred_keywords = self._extract_title_keywords(preferred.lower())
                            if preferred_keywords and job_keywords:
                                overlap = len(preferred_keywords.intersection(job_keywords))
                                max_possible = min(len(preferred_keywords), len(job_keywords))
                                if max_possible > 0:
                                    score = overlap / max_possible
                                    if self._has_role_match(preferred_keywords, job_keywords):
                                        score = min(1.0, score + 0.3)
                                    best_score = max(best_score, score)
                
                if best_score > 0:
                    return max(0.3, best_score)  # Minimum 0.3 if there's any match
        
        # Then try target_titles from resume template if available
        if resume_template and isinstance(resume_template, dict):
            target_titles = resume_template.get('target_titles', [])
            if target_titles:
                for target in target_titles:
                    # Skip None or non-string targets
                    if not target or not isinstance(target, str):
                        continue
                    if target.lower() in job_title_lower or job_title_lower in target.lower():
                        return 1.0
                return 0.3  # Has target titles but no match
        
        # Try to get current title from resume template
        current_title = ""
        if resume_template and isinstance(resume_template, dict):
            current_title = resume_template.get('title', '')
        
        if current_title:
            try:
                # Use semantic embedding for title comparison
                job_embedding = self.embedder.encode(job_title)
                resume_title_embedding = self.embedder.encode(current_title)
                
                # Calculate semantic similarity
                semantic_score = float(cosine_similarity(
                    job_embedding.reshape(1, -1),
                    resume_title_embedding.reshape(1, -1)
                )[0, 0])
                
                # Also calculate keyword overlap
                resume_keywords = self._extract_title_keywords(current_title.lower())
                keyword_score = 0
                if resume_keywords and job_keywords:
                    overlap = len(resume_keywords.intersection(job_keywords))
                    max_possible = min(len(resume_keywords), len(job_keywords))
                    if max_possible > 0:
                        keyword_score = overlap / max_possible
                        if self._has_role_match(resume_keywords, job_keywords):
                            keyword_score = min(1.0, keyword_score + 0.3)
                
                # Weighted combination: 70% semantic, 30% keyword
                combined_score = (semantic_score * 0.7) + (keyword_score * 0.3)
                return max(0.3, combined_score)  # Minimum 0.3 if there's a title
                
            except Exception as e:
                logger.warning(f"Error in semantic title matching for resume: {e}")
                # Fall back to keyword matching
                resume_keywords = self._extract_title_keywords(current_title.lower())
                if resume_keywords and job_keywords:
                    overlap = len(resume_keywords.intersection(job_keywords))
                    max_possible = min(len(resume_keywords), len(job_keywords))
                    if max_possible > 0:
                        score = overlap / max_possible
                        if self._has_role_match(resume_keywords, job_keywords):
                            score = min(1.0, score + 0.3)
                        return max(0.3, score)
        
        # Default fallback - check for common role patterns
        return self._calculate_generic_title_score(job_title_lower)
    
    def _extract_title_keywords(self, title: str) -> set:
        """Extract meaningful keywords from a job title"""
        # Remove common filler words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'at', 'to', 'for', 'with', 'by'}
        
        # Common title abbreviations to expand
        abbreviations = {
            'sr': 'senior',
            'jr': 'junior',
            'mgr': 'manager',
            'dir': 'director',
            'vp': 'vice president',
            'svp': 'senior vice president',
            'evp': 'executive vice president',
            'ceo': 'chief executive officer',
            'cto': 'chief technology officer',
            'cfo': 'chief financial officer',
            'coo': 'chief operating officer',
            'ciso': 'chief information security officer',
            'cio': 'chief information officer'
        }
        
        # Split and clean
        words = title.replace('-', ' ').replace('/', ' ').split()
        keywords = set()
        
        for word in words:
            word = word.strip('(),.')
            # Expand abbreviations
            if word in abbreviations:
                keywords.add(abbreviations[word])
            elif word not in stop_words and len(word) > 2:
                keywords.add(word)
        
        return keywords
    
    def _has_role_match(self, resume_keywords: set, job_keywords: set) -> bool:
        """Check if key role words match between titles"""
        # Key role indicators
        role_words = {
            'manager', 'director', 'lead', 'senior', 'principal', 'staff',
            'specialist', 'analyst', 'engineer', 'developer', 'coordinator',
            'administrator', 'architect', 'consultant', 'supervisor', 'head',
            'chief', 'president', 'officer', 'executive', 'associate'
        }
        
        resume_roles = resume_keywords.intersection(role_words)
        job_roles = job_keywords.intersection(role_words)
        
        return len(resume_roles.intersection(job_roles)) > 0
    
    def _calculate_generic_title_score(self, job_title: str) -> float:
        """Calculate a generic score based on job title complexity"""
        # For common roles, give a moderate score
        common_roles = ['specialist', 'analyst', 'manager', 'coordinator', 'engineer', 'developer']
        
        for role in common_roles:
            if role in job_title:
                return 0.5  # Moderate match for common roles
        
        return 0.3  # Low default score
    
    def _requirement_met_by_experience(self, requirement: str, experience: List) -> bool:
        """Check if requirement is met by experience"""
        
        if not experience or not requirement:
            return False
        
        # Ensure requirement is a string
        if not isinstance(requirement, str):
            return False
        
        # Simple keyword matching for now
        req_lower = requirement.lower()
        for exp in experience:
            if isinstance(exp, dict):
                exp_text = ' '.join([
                    exp.get('title', ''),
                    exp.get('company', ''),
                    ' '.join(exp.get('responsibilities', []))
                ]).lower()
                
                # Check for keyword overlap
                req_words = set(req_lower.split())
                exp_words = set(exp_text.split())
                overlap = len(req_words & exp_words) / len(req_words)
                
                if overlap > 0.3:
                    return True
        
        return False
    
    def _education_requirement_met(self, requirement: Dict, education: List) -> bool:
        """Check if education requirement is met - basic check only, LLM makes final decision"""
        
        if not education:
            return False
        
        req_degree = requirement.get('degree', '').lower()
        req_field = requirement.get('field', '').lower()
        
        for edu in education:
            if isinstance(edu, dict):
                # Get the full degree string (e.g., "Bachelor of Business Administration")
                degree_str = edu.get('degree', '').lower()
                
                # Check if degree level matches (bachelor's, master's, etc.)
                degree_levels = ['bachelor', 'master', 'phd', 'doctorate', 'associate']
                has_degree_level = False
                for level in degree_levels:
                    if level in req_degree.lower() and level in degree_str:
                        has_degree_level = True
                        break
                
                # If degree level matches, check field if specified
                if has_degree_level:
                    # If no specific field required, degree level match is enough
                    if not req_field or req_field == 'any':
                        return True
                    
                    # Clean up the field requirement string
                    req_field_clean = req_field.replace(',', ' ').replace(' or ', ' ').lower()
                    
                    # Check for "related field" which means any reasonable match is OK
                    if 'related field' in req_field_clean:
                        # Extract the main fields mentioned
                        # e.g., "Finance, Accounting, Business Administration, or a related field"
                        # becomes checking for finance, accounting, business
                        field_parts = req_field_clean.replace('related field', '').replace('a ', '').strip()
                        
                        # Check if any of the main field keywords appear in the degree
                        main_fields = ['business', 'finance', 'accounting', 'administration', 'management', 
                                      'economics', 'commerce', 'marketing']
                        for field in main_fields:
                            if field in req_field_clean and field in degree_str:
                                return True
                    
                    # Direct field matching - check if the field appears in the degree name
                    # Business Administration matches "Bachelor of Business Administration"
                    if 'business administration' in req_field_clean and 'business administration' in degree_str:
                        return True
                    if 'business' in req_field_clean and 'business' in degree_str:
                        return True
                    if 'accounting' in req_field_clean and 'accounting' in degree_str:
                        return True
                    if 'finance' in req_field_clean and 'finance' in degree_str:
                        return True
                    if 'computer' in req_field_clean and ('computer' in degree_str or 'software' in degree_str):
                        return True
                    if 'engineering' in req_field_clean and 'engineering' in degree_str:
                        return True
        
        return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        
        try:
            vec1 = np.array(vec1).reshape(1, -1)
            vec2 = np.array(vec2).reshape(1, -1)
            return float(cosine_similarity(vec1, vec2)[0][0])
        except:
            return 0.0
    
    def _parse_json_field(self, field: Any, default: Any = None) -> Any:
        """Parse JSON field safely"""
        
        if field is None:
            return default or {}
        
        if isinstance(field, str):
            try:
                return json.loads(field)
            except:
                return default or {}
        
        return field
    
    def _get_resumes(self, resume_ids: Optional[List[str]] = None) -> List[Dict]:
        """Get resumes from database"""
        
        if resume_ids:
            resumes = []
            for rid in resume_ids:
                resume = self.db.get_resume_by_id(rid)
                if resume:
                    resumes.append(resume)
            return resumes
        else:
            return self.db.get_all_resumes()
    
    def _get_jobs(self, job_ids: Optional[List[str]] = None) -> List[Dict]:
        """Get jobs from database"""
        
        if job_ids:
            jobs = []
            for jid in job_ids:
                job = self.db.get_job_by_id(jid)
                if job:
                    jobs.append(job)
            return jobs
        else:
            return self.db.get_all_jobs()
    
    def save_matches_to_db(self, matches: List[DetailedMatch]) -> int:
        """Save matches to database"""
        
        saved = 0
        for match in matches:
            try:
                match_data = {
                    'job_id': match.job_id,
                    'resume_id': match.resume_id,
                    'overall_score': match.overall_score,
                    'semantic_score': match.summary_to_description,
                    'title_match_score': match.title_alignment,
                    'skills_score': match.skills_match,
                    'experience_score': match.experience_match,
                    'education_score': match.education_match,  # Added
                    'location_score': match.location_match,    # Added
                    'salary_score': match.salary_alignment,    # Added
                    'requirements_score': match.experience_match,
                    'summary_to_description_score': match.summary_to_description,
                    'experience_to_requirements_score': match.experience_to_requirements,
                    'skills_to_skills_score': match.skills_to_skills,
                    'llm_score': match.llm_score,
                    'llm_assessment': match.llm_reasoning,
                    'match_reasons': match.llm_recommendations,
                    'skills_matched': match.matched_skills,
                    'skills_gap': match.missing_skills,
                    'requirements_matched': match.matched_requirements,
                    'requirements_gap': match.missing_requirements,
                    'location_preference_met': match.location_preference_met,  # Added
                    'remote_preference_met': match.remote_preference_met,       # Added
                    'salary_match': match.salary_match                         # Added
                }
                
                self.db.add_match(match_data)
                saved += 1
                
            except Exception as e:
                logger.error(f"Failed to save match: {e}")
        
        return saved


def enhanced_matching_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node for enhanced matching"""
    
    logger.info("Starting enhanced matching process")
    print("Starting enhanced matching process...")  # For process output
    
    # Get configuration
    config = state.get('config')
    
    # Convert PipelineConfig to dict if needed
    config_dict = {}
    top_k_matches = 1000  # Default value - essentially unlimited
    
    if config:
        if hasattr(config, 'to_dict'):
            config_dict = config.to_dict()
            top_k_matches = getattr(config, 'top_k_matches', 1000)
        elif hasattr(config, '__dict__'):
            config_dict = config.__dict__
            top_k_matches = config_dict.get('top_k_matches', 1000)
        elif isinstance(config, dict):
            config_dict = config
            top_k_matches = config.get('top_k_matches', 1000)
        else:
            # Config might be a string or other type, use defaults
            logger.warning(f"Unexpected config type: {type(config)}, using defaults")
            config_dict = {}
            top_k_matches = 1000
    
    print("Initializing matching engine...")
    
    # Initialize matcher
    matcher = EnhancedMatcher(config_dict)
    
    # Check if this is a manual match (clear all) or incremental match (from job search)
    clear_all_matches = config_dict.get('clear_all_matches', True)
    updated_job_ids = state.get('updated_job_ids', [])
    
    # Handle match clearing based on context
    if clear_all_matches:
        # Manual matching - clear all existing matches
        print("Clearing all previous matches...")
        matcher.db.clear_all_matches()
    elif updated_job_ids:
        # Job search - only clear matches for updated/new jobs
        print(f"Clearing matches for {len(updated_job_ids)} updated/new jobs...")
        for job_id in updated_job_ids:
            matcher.db.delete_matches_for_job(job_id)
    else:
        print("Preserving existing matches (incremental update)...")
    
    print(f"Performing matching analysis (top {top_k_matches} matches per resume)...")
    
    # Perform matching
    matches = matcher.match_all(
        resume_ids=state.get('resume_ids'),
        job_ids=state.get('job_ids'),
        top_k=top_k_matches
    )
    
    print(f"Found {len(matches)} potential matches")
    
    # Save to database
    print("Saving match results to database...")
    saved = matcher.save_matches_to_db(matches)
    logger.info(f"Saved {saved} matches to database")
    print(f"Successfully saved {saved} matches")
    
    # Convert matches to dict for state
    match_dicts = []
    for match in matches:
        match_dict = asdict(match)
        match_dicts.append(match_dict)
    
    # Update state - use best_matches to match pipeline expectations
    state['matches'] = match_dicts
    state['best_matches'] = match_dicts  # Pipeline expects this key
    state['match_count'] = len(matches)
    state['matching_results'] = match_dicts  # Also set this for compatibility
    
    logger.info(f"Completed matching: {len(matches)} matches found")
    
    return state

"""
Bidirectional Matching Node - Evaluates mutual fit between jobs and candidates

This module performs bidirectional matching between job postings and resumes,
evaluating both if the candidate fits the job AND if the job fits the candidate's preferences.
"""

from __future__ import annotations
import os
import json
import logging
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# Embeddings
from sklearn.metrics.pairwise import cosine_similarity
from graph.embeddings import JobEmbeddings, get_embeddings

# Universal template
from graph.vocation_template import (
    VocationTemplate,
    BidirectionalMatch,
    JobSeekerPreferences,
    WorkArrangement,
    EmploymentType
)

# LLM client
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Validation
from pydantic import BaseModel, Field


# -------------------------
# Configuration
# -------------------------

class BidirectionalMatchConfig:
    """Configuration for bidirectional matching"""
    
    # Embedding model - JobBERT-v2 for job-specific semantic understanding
    EMBEDDING_MODEL: str = "TechWolf/JobBERT-v2"
    
    # LLM settings
    DEFAULT_LLM_MODEL: str = "qwen3-4b-instruct-2507-f16"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 10000
    LLM_TIMEOUT: int = 180  # Increased for bidirectional assessment
    DEFAULT_BASE_URL: str = "http://localhost:8000/v1"
    USE_LLM_MATCHING: bool = True
    
    # Matching thresholds
    MIN_MATCH_SCORE: float = 0.3  # Minimum score to consider a match
    TOP_K_MATCHES: int = 10  # Number of top matches to return
    
    # Scoring weights for job->candidate fit
    JOB_WEIGHT_SKILLS: float = 0.35
    JOB_WEIGHT_EXPERIENCE: float = 0.25
    JOB_WEIGHT_EDUCATION: float = 0.15
    JOB_WEIGHT_SEMANTIC: float = 0.25
    
    # Scoring weights for candidate->job fit
    CANDIDATE_WEIGHT_COMPENSATION: float = 0.25
    CANDIDATE_WEIGHT_LOCATION: float = 0.20
    CANDIDATE_WEIGHT_CULTURE: float = 0.20
    CANDIDATE_WEIGHT_GROWTH: float = 0.15
    CANDIDATE_WEIGHT_TITLE: float = 0.20
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BidirectionalMatchConfig':
        """Create config from dictionary"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# -------------------------
# LLM Assessment
# -------------------------

class LLMFitAssessment(BaseModel):
    """LLM assessment of bidirectional fit"""
    job_fit_score: float = Field(ge=0, le=1)
    candidate_fit_score: float = Field(ge=0, le=1)
    job_fit_reasons: List[str] = Field(max_length=5)
    candidate_fit_reasons: List[str] = Field(max_length=5)
    concerns: List[str] = Field(max_length=3)
    recommendations: List[str] = Field(max_length=3)


class BidirectionalMatcher:
    """Performs bidirectional matching between jobs and candidates"""
    
    def __init__(
        self,
        config: BidirectionalMatchConfig = None,
        embeddings: JobEmbeddings = None,
        llm_client: OpenAI = None
    ):
        self.config = config or BidirectionalMatchConfig()
        self.embeddings = embeddings or get_embeddings()
        
        if self.config.USE_LLM_MATCHING and not llm_client:
            self.llm_client = OpenAI(
                base_url=os.getenv("OPENAI_BASE_URL", self.config.DEFAULT_BASE_URL),
                api_key=os.getenv("OPENAI_API_KEY", "dummy-key")
            )
        else:
            self.llm_client = llm_client
        
        self.model = os.getenv("LLM_MODEL", self.config.DEFAULT_LLM_MODEL)
    
    def match_templates(
        self,
        job_template: VocationTemplate,
        candidate_template: VocationTemplate,
        preferences: Optional[JobSeekerPreferences] = None
    ) -> BidirectionalMatch:
        """Perform bidirectional matching between job and candidate templates"""
        
        # Initialize match result
        match = BidirectionalMatch(
            job_template=job_template,
            candidate_template=candidate_template
        )
        
        # Calculate job->candidate fit (Is this candidate right for the job?)
        job_fit_scores = self._evaluate_job_fit(job_template, candidate_template)
        match.job_fit_score = job_fit_scores["overall"]
        match.job_fit_reasons = job_fit_scores["reasons"]
        match.missing_requirements = job_fit_scores["missing"]
        match.exceeds_requirements = job_fit_scores["exceeds"]
        
        # Calculate candidate->job fit (Is this job right for the candidate?)
        candidate_fit_scores = self._evaluate_candidate_fit(
            job_template, candidate_template, preferences
        )
        match.candidate_fit_score = candidate_fit_scores["overall"]
        match.candidate_fit_reasons = candidate_fit_scores["reasons"]
        match.unmet_preferences = candidate_fit_scores["unmet"]
        match.preference_matches = candidate_fit_scores["matches"]
        
        # Calculate detailed alignment scores
        match.skill_alignment = job_fit_scores["skill_score"]
        match.experience_alignment = job_fit_scores["experience_score"]
        match.education_alignment = job_fit_scores["education_score"]
        match.compensation_alignment = candidate_fit_scores["compensation_score"]
        match.location_alignment = candidate_fit_scores["location_score"]
        match.culture_alignment = candidate_fit_scores["culture_score"]
        
        # Calculate mutual fit score (geometric mean for balance)
        match.mutual_fit_score = np.sqrt(match.job_fit_score * match.candidate_fit_score)
        
        # Determine recommendation
        if match.mutual_fit_score >= 0.7:
            match.recommendation = "strong_match"
        elif match.mutual_fit_score >= 0.5:
            match.recommendation = "good_match"
        elif match.mutual_fit_score >= 0.3:
            match.recommendation = "partial_match"
        else:
            match.recommendation = "poor_match"
        
        # Add LLM assessment if enabled
        if self.config.USE_LLM_MATCHING and self.llm_client:
            llm_assessment = self._get_llm_assessment(job_template, candidate_template, preferences)
            if llm_assessment:
                # Blend LLM scores with calculated scores
                match.job_fit_score = 0.7 * match.job_fit_score + 0.3 * llm_assessment.job_fit_score
                match.candidate_fit_score = 0.7 * match.candidate_fit_score + 0.3 * llm_assessment.candidate_fit_score
                match.mutual_fit_score = np.sqrt(match.job_fit_score * match.candidate_fit_score)
                
                # Add LLM insights
                match.job_fit_reasons.extend(llm_assessment.job_fit_reasons[:2])
                match.candidate_fit_reasons.extend(llm_assessment.candidate_fit_reasons[:2])
                match.action_items = llm_assessment.recommendations
        
        # Generate action items based on match quality
        if not match.action_items:
            match.action_items = self._generate_action_items(match)
        
        return match
    
    def _evaluate_job_fit(
        self,
        job_template: VocationTemplate,
        candidate_template: VocationTemplate
    ) -> Dict[str, Any]:
        """Evaluate if candidate fits the job requirements"""
        
        results = {
            "overall": 0.0,
            "skill_score": 0.0,
            "experience_score": 0.0,
            "education_score": 0.0,
            "semantic_score": 0.0,
            "reasons": [],
            "missing": [],
            "exceeds": []
        }
        
        # 1. Skill matching with partial matching support
        job_skills = {s.skill_name.lower() for s in job_template.technical_skills + job_template.tools_technologies}
        candidate_skills = {s.skill_name.lower() for s in candidate_template.technical_skills + candidate_template.tools_technologies}
        
        # Function to check if any candidate skill contains or is contained by job skill
        def skills_match(job_skill, candidate_skills_set):
            job_skill_lower = job_skill.lower()
            for cand_skill in candidate_skills_set:
                # Check both directions: "ADP" in "ADP Workforce Now" or vice versa
                if job_skill_lower in cand_skill or cand_skill in job_skill_lower:
                    return True
                # Also check word-level matches: "GAAP compliance" matches "GAAP"
                job_words = set(job_skill_lower.split())
                cand_words = set(cand_skill.split())
                if job_words & cand_words:  # Any common words
                    return True
            return False
        
        required_skills = {s.skill_name.lower() for s in job_template.technical_skills if s.is_mandatory}
        
        # Count matches with partial matching
        matched_job_skills = set()
        matched_required = set()
        for job_skill in job_skills:
            if skills_match(job_skill, candidate_skills):
                matched_job_skills.add(job_skill)
                if job_skill in required_skills:
                    matched_required.add(job_skill)
        
        skill_overlap = matched_job_skills
        missing_required = required_skills - matched_required
        
        if required_skills:
            results["skill_score"] = len(matched_required) / len(required_skills)
        else:
            results["skill_score"] = len(matched_job_skills) / max(len(job_skills), 1)
        
        if results["skill_score"] >= 0.8:
            results["reasons"].append(f"Strong skill match ({len(skill_overlap)}/{len(job_skills)} skills)")
        elif results["skill_score"] >= 0.5:
            results["reasons"].append(f"Good skill match ({len(skill_overlap)}/{len(job_skills)} skills)")
        
        if missing_required:
            results["missing"].extend([f"Missing skill: {s}" for s in list(missing_required)[:3]])
        
        extra_skills = candidate_skills - job_skills
        if len(extra_skills) > 3:
            results["exceeds"].append(f"Has {len(extra_skills)} additional relevant skills")
        
        # 2. Experience matching
        if job_template.total_years_experience and candidate_template.total_years_experience:
            exp_diff = candidate_template.total_years_experience - job_template.total_years_experience
            if exp_diff >= 0:
                results["experience_score"] = min(1.0, 1.0 - abs(exp_diff) / 10)
                if exp_diff > 2:
                    results["exceeds"].append(f"Exceeds experience by {exp_diff:.1f} years")
            else:
                results["experience_score"] = max(0, 1.0 + exp_diff / 5)
                if exp_diff < -1:
                    results["missing"].append(f"Lacks {abs(exp_diff):.1f} years experience")
        else:
            results["experience_score"] = 0.5  # Neutral if not specified
        
        if results["experience_score"] >= 0.8:
            results["reasons"].append("Experience level matches requirements")
        
        # 3. Education matching
        if job_template.education_requirements:
            edu_matched = False
            for job_edu in job_template.education_requirements:
                if not job_edu.is_mandatory:
                    continue
                for cand_edu in candidate_template.education_requirements:
                    if self._education_matches(job_edu.degree_level, cand_edu.degree_level):
                        edu_matched = True
                        break
            results["education_score"] = 1.0 if edu_matched else 0.3
            
            if not edu_matched and any(e.is_mandatory for e in job_template.education_requirements):
                results["missing"].append("Missing required education level")
        else:
            results["education_score"] = 1.0  # No education requirements
        
        # 4. Semantic similarity
        job_text = f"{job_template.title} {job_template.summary} {' '.join(job_template.key_responsibilities)}"
        candidate_text = f"{candidate_template.title} {candidate_template.summary} {' '.join(candidate_template.achievements)}"
        
        job_embedding = self.embeddings.encode([job_text])
        candidate_embedding = self.embeddings.encode([candidate_text])
        results["semantic_score"] = float(cosine_similarity(job_embedding, candidate_embedding)[0, 0])
        
        if results["semantic_score"] >= 0.7:
            results["reasons"].append(f"High semantic match ({results['semantic_score']:.2f})")
        
        # Calculate overall score
        results["overall"] = (
            self.config.JOB_WEIGHT_SKILLS * results["skill_score"] +
            self.config.JOB_WEIGHT_EXPERIENCE * results["experience_score"] +
            self.config.JOB_WEIGHT_EDUCATION * results["education_score"] +
            self.config.JOB_WEIGHT_SEMANTIC * results["semantic_score"]
        )
        
        return results
    
    def _evaluate_candidate_fit(
        self,
        job_template: VocationTemplate,
        candidate_template: VocationTemplate,
        preferences: Optional[JobSeekerPreferences] = None
    ) -> Dict[str, Any]:
        """Evaluate if job fits the candidate's preferences"""
        
        results = {
            "overall": 0.0,
            "compensation_score": 0.0,
            "location_score": 0.0,
            "culture_score": 0.0,
            "growth_score": 0.0,
            "title_score": 0.0,
            "reasons": [],
            "unmet": [],
            "matches": []
        }
        
        # 1. Compensation alignment
        if candidate_template.compensation.minimum_salary and job_template.compensation.maximum_salary:
            if job_template.compensation.maximum_salary >= candidate_template.compensation.minimum_salary:
                salary_ratio = job_template.compensation.maximum_salary / max(candidate_template.compensation.minimum_salary, 1)
                results["compensation_score"] = min(1.0, salary_ratio)
                if salary_ratio >= 1.2:
                    results["matches"].append("Exceeds salary expectations")
                elif salary_ratio >= 1.0:
                    results["matches"].append("Meets salary expectations")
            else:
                results["compensation_score"] = 0.3
                results["unmet"].append("Below salary expectations")
        else:
            results["compensation_score"] = 0.5  # Neutral if not specified
        
        # Apply preference weight if available
        if preferences:
            results["compensation_score"] *= (0.5 + 0.5 * preferences.salary_importance)
        
        # 2. Location/remote alignment
        if candidate_template.location.work_arrangement and job_template.location.work_arrangement:
            if candidate_template.location.work_arrangement == job_template.location.work_arrangement:
                results["location_score"] = 1.0
                results["matches"].append(f"Work arrangement matches ({job_template.location.work_arrangement.value})")
            elif candidate_template.location.work_arrangement == WorkArrangement.FLEXIBLE:
                results["location_score"] = 0.8
            elif (candidate_template.location.work_arrangement == WorkArrangement.REMOTE and 
                  job_template.location.work_arrangement == WorkArrangement.HYBRID):
                results["location_score"] = 0.6
                results["unmet"].append("Prefers fully remote")
            else:
                results["location_score"] = 0.3
                results["unmet"].append("Work arrangement mismatch")
        else:
            results["location_score"] = 0.7  # Neutral
        
        if preferences:
            results["location_score"] *= (0.5 + 0.5 * preferences.location_importance)
        
        # 3. Culture/company fit
        culture_score = 0.5  # Base score
        
        if preferences:
            # Check company preferences
            company_name = job_template.metadata.get("company", "").lower()
            if preferences.avoid_companies and any(c.lower() in company_name for c in preferences.avoid_companies):
                culture_score = 0.1
                results["unmet"].append("Company on avoid list")
            elif preferences.preferred_companies and any(c.lower() in company_name for c in preferences.preferred_companies):
                culture_score = 1.0
                results["matches"].append("Preferred company")
            
            # Check industry preferences
            if preferences.preferred_industries and job_template.culture_fit.industry_preferences:
                industry_overlap = set(preferences.preferred_industries) & set(job_template.culture_fit.industry_preferences)
                if industry_overlap:
                    culture_score = max(culture_score, 0.8)
                    results["matches"].append("Industry match")
            
            # Apply importance weight
            culture_score *= (0.5 + 0.5 * preferences.company_culture_importance)
        
        results["culture_score"] = culture_score
        
        # 4. Growth opportunities
        growth_score = 0.5
        if job_template.culture_fit.career_growth_importance:
            if "high" in str(job_template.culture_fit.career_growth_importance).lower():
                growth_score = 0.9
                results["matches"].append("Strong growth opportunities")
        
        if preferences:
            growth_score *= (0.5 + 0.5 * preferences.career_growth_importance)
        
        results["growth_score"] = growth_score
        
        # 5. Title/role alignment
        title_score = 0.5
        job_title_lower = job_template.title.lower()
        candidate_title_lower = candidate_template.title.lower()
        
        if preferences and preferences.desired_titles:
            if any(t.lower() in job_title_lower for t in preferences.desired_titles):
                title_score = 1.0
                results["matches"].append("Desired job title")
        elif preferences and preferences.avoid_titles:
            if any(t.lower() in job_title_lower for t in preferences.avoid_titles):
                title_score = 0.1
                results["unmet"].append("Undesired job title")
        else:
            # Basic title similarity
            title_words = set(job_title_lower.split())
            candidate_words = set(candidate_title_lower.split())
            if title_words & candidate_words:
                title_score = 0.7
        
        results["title_score"] = title_score
        
        # Calculate overall candidate fit score
        results["overall"] = (
            self.config.CANDIDATE_WEIGHT_COMPENSATION * results["compensation_score"] +
            self.config.CANDIDATE_WEIGHT_LOCATION * results["location_score"] +
            self.config.CANDIDATE_WEIGHT_CULTURE * results["culture_score"] +
            self.config.CANDIDATE_WEIGHT_GROWTH * results["growth_score"] +
            self.config.CANDIDATE_WEIGHT_TITLE * results["title_score"]
        )
        
        # Check deal breakers
        if preferences and preferences.deal_breakers:
            job_text = f"{job_template.title} {job_template.summary}".lower()
            for deal_breaker in preferences.deal_breakers:
                if deal_breaker.lower() in job_text:
                    results["overall"] *= 0.3  # Severe penalty
                    results["unmet"].append(f"Deal breaker: {deal_breaker}")
        
        return results
    
    def _education_matches(self, required: str, candidate: str) -> bool:
        """Check if candidate education meets requirements"""
        education_levels = {
            "high school": 1,
            "associate": 2,
            "bachelor": 3,
            "master": 4,
            "phd": 5,
            "doctorate": 5
        }
        
        req_level = 0
        cand_level = 0
        
        for edu, level in education_levels.items():
            if edu in required.lower():
                req_level = level
            if edu in candidate.lower():
                cand_level = level
        
        return cand_level >= req_level
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _get_llm_assessment(
        self,
        job_template: VocationTemplate,
        candidate_template: VocationTemplate,
        preferences: Optional[JobSeekerPreferences] = None
    ) -> Optional[LLMFitAssessment]:
        """Get LLM assessment of bidirectional fit"""
        
        try:
            system_prompt = "Assess job-candidate fit bidirectionally. Return JSON only."
            
            # Prepare concise representations
            job_summary = {
                "title": job_template.title,
                "required_skills": [s.skill_name for s in job_template.technical_skills if s.is_mandatory][:5],
                "nice_to_have": [s.skill_name for s in job_template.technical_skills if not s.is_mandatory][:3],
                "experience": job_template.total_years_experience,
                "salary": f"${job_template.compensation.minimum_salary}-${job_template.compensation.maximum_salary}" if job_template.compensation.maximum_salary else "Not specified",
                "location": job_template.location.work_arrangement.value if job_template.location.work_arrangement else "Not specified"
            }
            
            candidate_summary = {
                "current_title": candidate_template.title,
                "skills": [s.skill_name for s in candidate_template.technical_skills][:8],
                "experience": candidate_template.total_years_experience,
                "salary_expectation": candidate_template.compensation.minimum_salary if candidate_template.compensation.minimum_salary else "Flexible",
                "work_preference": candidate_template.location.work_arrangement.value if candidate_template.location.work_arrangement else "Flexible"
            }
            
            user_prompt = f"""Assess this match:

JOB: {json.dumps(job_summary)}
CANDIDATE: {json.dumps(candidate_summary)}

Return JSON:
{{
  "job_fit_score": 0.75,
  "candidate_fit_score": 0.65,
  "job_fit_reasons": ["Has required skills", "Meets experience"],
  "candidate_fit_reasons": ["Good salary", "Remote work available"],
  "concerns": ["May be overqualified"],
  "recommendations": ["Discuss growth path", "Clarify remote policy"]
}}"""
            
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
                timeout=self.config.LLM_TIMEOUT
            )
            
            content = response.choices[0].message.content
            data = json.loads(content.strip().strip("```json").strip("```"))
            
            return LLMFitAssessment(**data)
            
        except Exception as e:
            logging.warning(f"LLM assessment failed: {e}")
            return None
    
    def _generate_action_items(self, match: BidirectionalMatch) -> List[str]:
        """Generate action items based on match quality"""
        
        items = []
        
        if match.recommendation == "strong_match":
            items.append("Schedule interview immediately")
            items.append("Prepare competitive offer")
        elif match.recommendation == "good_match":
            items.append("Schedule screening call")
            if match.missing_requirements:
                items.append("Assess ability to learn missing skills")
        elif match.recommendation == "partial_match":
            items.append("Review with hiring manager")
            if match.unmet_preferences:
                items.append("Consider adjusting job parameters")
        
        if match.exceeds_requirements and match.candidate_fit_score < 0.6:
            items.append("Discuss growth opportunities and challenges")
        
        if match.compensation_alignment < 0.5:
            items.append("Review compensation package")
        
        return items[:3]  # Top 3 items


# -------------------------
# Batch Matching
# -------------------------

def perform_batch_matching(
    job_templates: List[VocationTemplate],
    candidate_templates: List[VocationTemplate],
    preferences_map: Optional[Dict[str, JobSeekerPreferences]] = None,
    config: Optional[BidirectionalMatchConfig] = None,
    top_k: int = 10
) -> List[BidirectionalMatch]:
    """Perform batch bidirectional matching between multiple jobs and candidates"""
    
    matcher = BidirectionalMatcher(config=config)
    all_matches = []
    
    for candidate_template in candidate_templates:
        candidate_id = candidate_template.source_id
        preferences = preferences_map.get(candidate_id) if preferences_map else None
        
        candidate_matches = []
        for job_template in job_templates:
            match = matcher.match_templates(job_template, candidate_template, preferences)
            candidate_matches.append(match)
        
        # Sort by mutual fit score
        candidate_matches.sort(key=lambda m: m.mutual_fit_score, reverse=True)
        
        # Keep top K matches
        all_matches.extend(candidate_matches[:top_k])
    
    # Sort all matches by mutual fit
    all_matches.sort(key=lambda m: m.mutual_fit_score, reverse=True)
    
    return all_matches


# -------------------------
# LangGraph Node Interface
# -------------------------

def bidirectional_matching_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible node for bidirectional matching.
    
    Expected state keys:
        - job_templates (list): List of VocationTemplate objects for jobs
        - candidate_templates (list): List of VocationTemplate objects for candidates
        - preferences (dict, optional): Map of candidate_id to JobSeekerPreferences
        - config (dict, optional): Configuration overrides
        - top_k (int, optional): Number of top matches per candidate
    
    Returns state with:
        - bidirectional_matches: List of BidirectionalMatch dictionaries
        - best_matches: Top matches formatted for display
        - diagnostics: Matching diagnostics
    """
    
    # Set up logging
    if state.get("verbose", False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Get configuration
    config = BidirectionalMatchConfig()
    if "config" in state and isinstance(state["config"], dict):
        config = BidirectionalMatchConfig.from_dict(state["config"])
    
    try:
        # Get templates
        job_templates = state.get("job_templates", [])
        candidate_templates = state.get("candidate_templates", [])
        preferences_map = state.get("preferences", {})
        top_k = state.get("top_k", config.TOP_K_MATCHES)
        
        if not job_templates or not candidate_templates:
            logging.warning("No templates to match")
            state["bidirectional_matches"] = []
            state["best_matches"] = []
            return state
        
        logging.info(f"Matching {len(candidate_templates)} candidates with {len(job_templates)} jobs")
        
        # Perform matching
        matches = perform_batch_matching(
            job_templates,
            candidate_templates,
            preferences_map,
            config,
            top_k
        )
        
        # Convert to dictionaries for state
        bidirectional_matches = []
        best_matches = []
        
        for match in matches:
            # Create display-friendly format
            best_match = {
                "candidate_name": match.candidate_template.metadata.get("name", "Unknown"),
                "candidate_email": match.candidate_template.metadata.get("email", ""),
                "job_title": match.job_template.title,
                "company": match.job_template.metadata.get("company", "Unknown"),
                "mutual_fit_score": match.mutual_fit_score,
                "job_fit_score": match.job_fit_score,
                "candidate_fit_score": match.candidate_fit_score,
                "recommendation": match.recommendation,
                "job_fit_reasons": match.job_fit_reasons[:3],
                "candidate_fit_reasons": match.candidate_fit_reasons[:3],
                "missing_requirements": match.missing_requirements[:3],
                "unmet_preferences": match.unmet_preferences[:3],
                "action_items": match.action_items,
                "alignment_scores": {
                    "skills": match.skill_alignment,
                    "experience": match.experience_alignment,
                    "education": match.education_alignment,
                    "compensation": match.compensation_alignment,
                    "location": match.location_alignment,
                    "culture": match.culture_alignment
                }
            }
            best_matches.append(best_match)
            
            # Store full match data (excluding template objects for serialization)
            match_dict = {
                "job_id": match.job_template.source_id,
                "candidate_id": match.candidate_template.source_id,
                "mutual_fit_score": match.mutual_fit_score,
                "job_fit_score": match.job_fit_score,
                "candidate_fit_score": match.candidate_fit_score,
                "recommendation": match.recommendation,
                "job_fit_reasons": match.job_fit_reasons,
                "candidate_fit_reasons": match.candidate_fit_reasons,
                "missing_requirements": match.missing_requirements,
                "exceeds_requirements": match.exceeds_requirements,
                "unmet_preferences": match.unmet_preferences,
                "preference_matches": match.preference_matches,
                "action_items": match.action_items,
                "skill_alignment": match.skill_alignment,
                "experience_alignment": match.experience_alignment,
                "education_alignment": match.education_alignment,
                "compensation_alignment": match.compensation_alignment,
                "location_alignment": match.location_alignment,
                "culture_alignment": match.culture_alignment,
                "metadata": match.metadata
            }
            bidirectional_matches.append(match_dict)
        
        # Update state
        state["bidirectional_matches"] = bidirectional_matches
        state["best_matches"] = best_matches[:50]  # Top 50 for display
        
        if "diagnostics" not in state:
            state["diagnostics"] = {}
        
        state["diagnostics"]["bidirectional_matching"] = {
            "candidates_evaluated": len(candidate_templates),
            "jobs_evaluated": len(job_templates),
            "total_matches": len(matches),
            "strong_matches": sum(1 for m in matches if m.recommendation == "strong_match"),
            "good_matches": sum(1 for m in matches if m.recommendation == "good_match"),
            "partial_matches": sum(1 for m in matches if m.recommendation == "partial_match"),
            "poor_matches": sum(1 for m in matches if m.recommendation == "poor_match")
        }
        
        logging.info(f"Matching complete: {len(matches)} matches found")
        
    except Exception as e:
        logging.error(f"Bidirectional matching failed: {e}")
        import traceback
        traceback.print_exc()
        state["bidirectional_matches"] = []
        state["best_matches"] = []
        state["diagnostics"] = {
            "bidirectional_matching": {
                "status": "error",
                "error": str(e)
            }
        }
    
    return state


# -------------------------
# Standalone Testing
# -------------------------

if __name__ == "__main__":
    # Test bidirectional matching
    import sys
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("Testing bidirectional matching...")
    print("-" * 60)
    
    # Create sample templates
    from graph.vocation_template import SkillRequirement, ExperienceRequirement, EducationRequirement
    
    # Sample job template
    job_template = VocationTemplate(
        source_type="job_posting",
        source_id="job_001",
        title="Senior Payroll Specialist",
        summary="Looking for experienced payroll specialist"
    )
    job_template.technical_skills = [
        SkillRequirement("ADP", "advanced", 3, True, ["ADP experience required"], 0.9),
        SkillRequirement("Payroll Processing", "expert", 5, True, ["5+ years payroll"], 0.9),
        SkillRequirement("Excel", "advanced", None, False, ["Excel preferred"], 0.7)
    ]
    job_template.total_years_experience = 5
    job_template.compensation.minimum_salary = 60000
    job_template.compensation.maximum_salary = 80000
    job_template.location.work_arrangement = WorkArrangement.HYBRID
    
    # Sample candidate template
    candidate_template = VocationTemplate(
        source_type="resume",
        source_id="candidate_001",
        title="Senior Payroll Coordinator",
        summary="7+ years payroll experience with ADP"
    )
    candidate_template.technical_skills = [
        SkillRequirement("ADP", "expert", None, False, [], 1.0),
        SkillRequirement("Payroll Processing", "expert", None, False, [], 1.0),
        SkillRequirement("NetSuite", "advanced", None, False, [], 0.9)
    ]
    candidate_template.total_years_experience = 7
    candidate_template.compensation.minimum_salary = 70000
    candidate_template.location.work_arrangement = WorkArrangement.REMOTE
    
    # Sample preferences
    preferences = JobSeekerPreferences(
        desired_titles=["Payroll Specialist", "Payroll Manager"],
        preferred_work_arrangement=WorkArrangement.REMOTE,
        minimum_acceptable_salary=65000,
        salary_importance=0.8,
        location_importance=0.9,
        work_life_balance_importance=0.7
    )
    
    # Perform matching
    matcher = BidirectionalMatcher()
    match = matcher.match_templates(job_template, candidate_template, preferences)
    
    # Display results
    print("\n" + "=" * 60)
    print("BIDIRECTIONAL MATCH RESULTS")
    print("=" * 60)
    
    print(f"\nMutual Fit Score: {match.mutual_fit_score:.2f}")
    print(f"Recommendation: {match.recommendation.upper()}")
    
    print(f"\n--- Job → Candidate Fit: {match.job_fit_score:.2f} ---")
    print("Reasons:")
    for reason in match.job_fit_reasons:
        print(f"  {reason}")
    if match.missing_requirements:
        print("Missing:")
        for missing in match.missing_requirements:
            print(f"  ✗ {missing}")
    if match.exceeds_requirements:
        print("Exceeds:")
        for exceeds in match.exceeds_requirements:
            print(f"  ↑ {exceeds}")
    
    print(f"\n--- Candidate → Job Fit: {match.candidate_fit_score:.2f} ---")
    print("Matches:")
    for match_item in match.preference_matches:
        print(f"  {match_item}")
    if match.unmet_preferences:
        print("Unmet Preferences:")
        for unmet in match.unmet_preferences:
            print(f"  ✗ {unmet}")
    
    print("\n--- Alignment Scores ---")
    print(f"  Skills: {match.skill_alignment:.2f}")
    print(f"  Experience: {match.experience_alignment:.2f}")
    print(f"  Education: {match.education_alignment:.2f}")
    print(f"  Compensation: {match.compensation_alignment:.2f}")
    print(f"  Location: {match.location_alignment:.2f}")
    print(f"  Culture: {match.culture_alignment:.2f}")
    
    print("\n--- Action Items ---")
    for item in match.action_items:
        print(f"  • {item}")
    
    print("\nTest completed successfully!")

"""
Job Matching Node - Compares resumes to job postings using JobBERT-v2 embeddings and LLM analysis
"""

from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

# Embeddings
from sklearn.metrics.pairwise import cosine_similarity
from graph.embeddings import JobEmbeddings, get_embeddings

# LLM client
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# Validation
from pydantic import BaseModel, Field


# -------------------------
# Configuration
# -------------------------

class JobMatchConfig:
    """Configuration for job matching"""
    
    def __init__(
        self,
        semantic_weight: float = None,
        skills_weight: float = None,
        experience_weight: float = None,
        llm_weight: float = None,
        min_score: float = None,
        top_k: int = None,
        **kwargs
    ):
        # Embedding model - JobBERT-v2 for job-specific semantic understanding
        self.EMBEDDING_MODEL: str = "TechWolf/JobBERT-v2"
        
        # LLM settings for llama.cpp
        self.DEFAULT_LLM_MODEL: str = "qwen3-4b-instruct-2507-f16"
        self.LLM_TEMPERATURE: float = 0.1
        self.LLM_MAX_TOKENS: int = 10000
        self.LLM_TIMEOUT: int = 180  # Increased for matching assessment
        self.DEFAULT_BASE_URL: str = "http://localhost:8000/v1"
        
        # Matching thresholds
        self.MIN_SEMANTIC_SCORE: float = min_score if min_score is not None else 0.3  # Minimum similarity to consider
        self.TOP_K_MATCHES: int = top_k if top_k is not None else 1000  # Number of top matches to return (essentially unlimited)
        
        # Scoring weights
        self.WEIGHT_SEMANTIC: float = semantic_weight if semantic_weight is not None else 0.4  # Weight for embedding similarity
        self.WEIGHT_SKILLS: float = skills_weight if skills_weight is not None else 0.3   # Weight for skills match
        self.WEIGHT_EXPERIENCE: float = experience_weight if experience_weight is not None else 0.2  # Weight for experience match
        self.WEIGHT_LLM: float = llm_weight if llm_weight is not None else 0.1      # Weight for LLM assessment
        
        # Apply any additional kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'JobMatchConfig':
        """Create config from dictionary"""
        return cls(**config_dict)


# -------------------------
# Data Models
# -------------------------

@dataclass
class JobMatch:
    """Single job-resume match result"""
    job_id: str
    job_title: Optional[str]
    company: Optional[str]
    semantic_score: float
    skills_score: float
    experience_score: float
    llm_score: float
    overall_score: float
    match_reasons: List[str]
    missing_skills: List[str]
    llm_assessment: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingResult:
    """Complete matching results for a resume"""
    resume_id: str
    candidate_name: Optional[str]
    total_jobs_evaluated: int
    matches: List[JobMatch]
    processing_time: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMMatchAssessment(BaseModel):
    """Validated LLM assessment of job match"""
    fit_score: float = Field(ge=0, le=1)
    key_strengths: List[str] = Field(max_length=5)
    gaps: List[str] = Field(max_length=5)
    recommendation: str = Field(max_length=500)
    confidence: float = Field(ge=0, le=1)


# -------------------------
# Embedding Processor
# -------------------------

class JobBERTProcessor:
    """Handles JobBERT-v2 embeddings for job matching (wrapper for centralized embeddings)"""
    
    def __init__(self, llm_name: str = None):
        # Use centralized embeddings
        self.embeddings = get_embeddings()
        self.llm_name = self.embeddings.llm_name
        logging.info(f"Using centralized JobBERT-v2 embeddings: {self.llm_name}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode single text to embedding"""
        return self.embeddings.encode(text)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts to embeddings"""
        return self.embeddings.encode(texts, batch_size=batch_size)
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return self.embeddings.match_score(embedding1, embedding2)


# -------------------------
# Skills Matcher
# -------------------------

class SkillsMatcher:
    """Matches skills between resume and job posting"""
    
    @staticmethod
    def normalize_skill(skill: str) -> str:
        """Normalize skill for comparison"""
        return skill.lower().strip().replace('-', ' ').replace('_', ' ')
    
    @staticmethod
    def calculate_skills_overlap(
        resume_skills: List[str],
        job_text: str
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate skills overlap between resume and job"""
        
        if not resume_skills:
            return 0.0, [], []
        
        # Normalize resume skills
        normalized_resume = {SkillsMatcher.normalize_skill(s): s for s in resume_skills}
        job_text_lower = job_text.lower()
        
        # Find matching skills
        matched_skills = []
        for norm_skill, original_skill in normalized_resume.items():
            if norm_skill in job_text_lower:
                matched_skills.append(original_skill)
        
        # Calculate score
        score = len(matched_skills) / len(resume_skills) if resume_skills else 0
        
        # Identify missing skills (simplified - would need job skill extraction)
        missing_skills = []  # Would need to extract required skills from job
        
        return score, matched_skills, missing_skills
    
    @staticmethod
    def calculate_experience_match(
        resume_years: Optional[float],
        job_text: str
    ) -> float:
        """Calculate experience match score"""
        
        if resume_years is None:
            return 0.5  # Neutral score if unknown
        
        # Look for years requirements in job text
        import re
        years_patterns = [
            r'(\d+)\+?\s*years?',
            r'(\d+)-(\d+)\s*years?',
            r'minimum\s+(\d+)\s*years?'
        ]
        
        required_years = []
        for pattern in years_patterns:
            matches = re.findall(pattern, job_text.lower())
            for match in matches:
                if isinstance(match, tuple):
                    # Range pattern
                    required_years.extend([int(m) for m in match if m])
                else:
                    required_years.append(int(match))
        
        if not required_years:
            return 0.7  # Good score if no specific requirement
        
        # Compare against requirements
        min_required = min(required_years)
        max_required = max(required_years)
        
        if resume_years >= min_required:
            if resume_years <= max_required * 1.5:  # Not too overqualified
                return 1.0
            else:
                return 0.7  # Overqualified
        else:
            # Under-qualified - proportional score
            return max(0.3, resume_years / min_required)


# -------------------------
# LLM Match Assessor
# -------------------------

class LLMMatchAssessor:
    """Uses LLM to assess job-resume match quality"""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        config: JobMatchConfig = None
    ):
        self.config = config or JobMatchConfig()
        self.model = model or os.getenv("LLM_MODEL", self.config.DEFAULT_LLM_MODEL)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", self.config.DEFAULT_BASE_URL)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        logging.info(f"Initialized LLM assessor with model: {self.model}")
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def assess_match(
        self,
        resume_summary: str,
        job_summary: str,
        quick: bool = True
    ) -> LLMMatchAssessment:
        """Assess job-resume match using LLM"""
        
        if quick:
            # Quick assessment with minimal context
            return self._quick_assessment(resume_summary, job_summary)
        else:
            # Detailed assessment
            return self._detailed_assessment(resume_summary, job_summary)
    
    def _quick_assessment(
        self,
        resume_summary: str,
        job_summary: str
    ) -> LLMMatchAssessment:
        """Quick match assessment"""
        
        # Truncate for speed
        resume_summary = resume_summary[:500]
        job_summary = job_summary[:500]
        
        system_prompt = "Assess job-resume fit. Return JSON only."
        
        user_prompt = f"""Compare:

RESUME: {resume_summary}

JOB: {job_summary}

Return JSON:
{{
  "fit_score": 0.0-1.0,
  "key_strengths": ["strength1", "strength2"],
  "gaps": ["ONLY list skills/requirements explicitly mentioned in the JOB that the candidate lacks"],
  "recommendation": "brief assessment",
  "confidence": 0.0-1.0
}}

IMPORTANT: Only list gaps for skills that are EXPLICITLY mentioned in the job description. Do NOT invent or assume missing skills."""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=300,
                timeout=30
            )
            
            content = response.choices[0].message.content
            data = self._parse_json_response(content)
            
            return LLMMatchAssessment(
                fit_score=float(data.get("fit_score", 0.5)),
                key_strengths=data.get("key_strengths", [])[:5],
                gaps=data.get("gaps", [])[:5],
                recommendation=str(data.get("recommendation", ""))[:500],
                confidence=float(data.get("confidence", 0.5))
            )
            
        except Exception as e:
            logging.warning(f"LLM assessment failed: {e}")
            return LLMMatchAssessment(
                fit_score=0.5,
                key_strengths=[],
                gaps=[],
                recommendation="Unable to assess",
                confidence=0.1
            )
    
    def _detailed_assessment(
        self,
        resume_summary: str,
        job_summary: str
    ) -> LLMMatchAssessment:
        """Detailed match assessment (not implemented for brevity)"""
        return self._quick_assessment(resume_summary, job_summary)
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM JSON response"""
        import re
        
        response = response.strip()
        
        # Remove markdown
        if "```" in response:
            response = re.sub(r'```(?:json)?', '', response)
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        
        return {}


# -------------------------
# Job Matching Pipeline
# -------------------------

class JobMatchingPipeline:
    """Main pipeline for matching resumes to jobs"""
    
    def __init__(
        self,
        config: JobMatchConfig = None,
        embedding_processor: JobBERTProcessor = None,
        llm_assessor: LLMMatchAssessor = None
    ):
        self.config = config or JobMatchConfig()
        self.embedding_processor = embedding_processor or JobBERTProcessor()
        self.llm_assessor = llm_assessor or LLMMatchAssessor(config=self.config)
        self.skills_matcher = SkillsMatcher()
    
    def match_resume_to_jobs(
        self,
        resume_template: Dict[str, Any],
        jobs: List[Dict[str, Any]],
        use_llm: bool = True,
        verbose: bool = False
    ) -> MatchingResult:
        """Match a resume against multiple job postings"""
        
        start_time = datetime.now()
        
        # Extract resume info
        resume_text = self._create_resume_text(resume_template)
        raw_skills = resume_template.get("requirements_match", {}).get("technical_skills", [])
        
        # Clean up resume skills - ensure they're strings
        resume_skills = []
        for skill in raw_skills:
            if isinstance(skill, dict):
                skill_name = skill.get('skill_name', skill.get('name', skill.get('skill', '')))
                if skill_name:
                    resume_skills.append(skill_name)
            elif isinstance(skill, str):
                resume_skills.append(skill)
        
        resume_years = resume_template.get("candidate_profile", {}).get("years_experience")
        
        # Try multiple places for candidate name
        candidate_name = (
            resume_template.get("candidate_profile", {}).get("name") or
            resume_template.get("contact_info", {}).get("name") or
            resume_template.get("personal_info", {}).get("name") or
            resume_template.get("basic_info", {}).get("name") or
            ""
        )
        
        # If name is empty, try to extract from job titles
        if not candidate_name or candidate_name == "":
            # Look for a job title that might indicate the person's role
            work_history = resume_template.get("work_history", [])
            if work_history and len(work_history) > 0:
                first_job = work_history[0]
                job_title = first_job.get("title", "")
                if job_title:
                    candidate_name = f"{job_title} Candidate"
                else:
                    candidate_name = "Experienced Professional"
            else:
                # Use field if available
                field = resume_template.get("candidate_profile", {}).get("detected_field", "")
                if field:
                    candidate_name = f"{field.title()} Professional"
                else:
                    candidate_name = "Professional Candidate"
        
        if verbose:
            logging.info(f"Matching resume for {candidate_name} against {len(jobs)} jobs")
        
        # Generate resume embedding
        resume_embedding = self.embedding_processor.encode_text(resume_text)
        
        # Process each job
        matches = []
        for i, job in enumerate(jobs):
            if verbose and i % 10 == 0:
                logging.info(f"Processing job {i+1}/{len(jobs)}")
            
            match = self._match_single_job(
                resume_embedding,
                resume_text,
                resume_skills,
                resume_years,
                resume_template,
                job,
                use_llm=use_llm and i < 5  # Only use LLM for top matches to save time
            )
            
            # Debug log the match score
            if verbose:
                logging.info(f"Job {i}: score={match.overall_score:.3f}, min_required={self.config.MIN_SEMANTIC_SCORE}")
            
            if match.overall_score >= self.config.MIN_SEMANTIC_SCORE:
                matches.append(match)
        
        # Sort by overall score
        matches.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Keep top K
        matches = matches[:self.config.TOP_K_MATCHES]
        
        # If we didn't use LLM for all top matches, do it now
        if use_llm:
            for match in matches[:5]:
                if match.llm_score == 0:
                    match = self._add_llm_assessment(
                        match,
                        resume_template,
                        next(j for j in jobs if self._get_job_id(j) == match.job_id)
                    )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return MatchingResult(
            resume_id=resume_template.get("candidate_profile", {}).get("email", "unknown"),
            candidate_name=candidate_name,
            total_jobs_evaluated=len(jobs),
            matches=matches,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    def _match_single_job(
        self,
        resume_embedding: np.ndarray,
        resume_text: str,
        resume_skills: List[str],
        resume_years: Optional[float],
        resume_template: Dict[str, Any],
        job: Dict[str, Any],
        use_llm: bool = False
    ) -> JobMatch:
        """Match resume against a single job"""
        
        # Get job text
        job_text = job.get("text", job.get("full_text", ""))
        if not job_text:
            job_text = json.dumps(job)
        
        # Check if job has VocationTemplate
        job_template = job.get("vocation_template", {})
        
        # If template is a JSON string, parse it
        if isinstance(job_template, str):
            try:
                job_template = json.loads(job_template) if job_template else {}
            except json.JSONDecodeError:
                logging.warning(f"Failed to parse job template JSON")
                job_template = {}
        
        # Debug logging
        logging.info(f"Job template type: {type(job_template)}, empty: {not job_template}")
        if job_template:
            logging.info(f"Template keys: {list(job_template.keys()) if isinstance(job_template, dict) else 'Not a dict'}")
        
        # Extract additional matching factors from template
        job_years_required = None
        job_location = None
        job_remote_policy = None
        job_salary_range = None
        
        if job_template and isinstance(job_template, dict):
            # Extract skills from template for better matching
            job_skills = []
            
            # Handle technical skills (list of dicts with 'skill_name' or 'name' field or strings)
            tech_skills = job_template.get("technical_skills", [])
            if tech_skills:
                if isinstance(tech_skills, list):
                    for skill in tech_skills:
                        if isinstance(skill, dict):
                            # Try different possible field names
                            skill_name = skill.get("skill_name") or skill.get("name") or str(skill)
                        else:
                            skill_name = str(skill)
                        if skill_name:
                            job_skills.append(skill_name)
                elif isinstance(tech_skills, str):
                    job_skills.extend([s.strip() for s in tech_skills.split(",") if s.strip()])
            
            # Handle tools/technologies
            tools = job_template.get("tools_technologies", [])
            if tools:
                if isinstance(tools, list):
                    for tool in tools:
                        if isinstance(tool, dict):
                            tool_name = tool.get("tool_name") or tool.get("name") or str(tool)
                        else:
                            tool_name = str(tool)
                        if tool_name:
                            job_skills.append(tool_name)
                elif isinstance(tools, str):
                    job_skills.extend([t.strip() for t in tools.split(",") if t.strip()])
            
            # Handle soft skills
            soft = job_template.get("soft_skills", [])
            if soft:
                if isinstance(soft, list):
                    for skill in soft:
                        if isinstance(skill, dict):
                            skill_name = skill.get("skill_name") or skill.get("name") or str(skill)
                        else:
                            skill_name = str(skill)
                        if skill_name:
                            job_skills.append(skill_name)
                elif isinstance(soft, str):
                    job_skills.extend([s.strip() for s in soft.split(",") if s.strip()])
            
            # Extract other factors for matching
            # Try different field names for experience
            job_years_required = (job_template.get("years_experience_required") or 
                                 job_template.get("total_years_experience") or
                                 job_template.get("experience_requirements", {}).get("minimum_years") if isinstance(job_template.get("experience_requirements"), dict) else None)
            job_location = job_template.get("location", "")
            job_remote_policy = job_template.get("remote_policy", "")
            
            # Get salary info - try multiple field structures
            compensation = job_template.get("compensation", {})
            if isinstance(compensation, dict):
                salary_min = compensation.get("min_salary") or compensation.get("salary_min")
                salary_max = compensation.get("max_salary") or compensation.get("salary_max")
            else:
                salary_min = job_template.get("salary_min")
                salary_max = job_template.get("salary_max")
            
            if salary_min or salary_max:
                job_salary_range = (salary_min, salary_max)
            
            # Use template for more accurate matching
            logging.info(f"Using VocationTemplate with {len(job_skills)} extracted skills: {job_skills[:5]}")
            logging.info(f"Template has: years_req={job_years_required}, location={job_location}, remote={job_remote_policy}")
        else:
            job_skills = []
            logging.info("No VocationTemplate found, using text extraction")
        
        # Generate job embedding
        job_embedding = self.embedding_processor.encode_text(job_text)
        
        # Calculate semantic similarity
        semantic_score = self.embedding_processor.compute_similarity(
            resume_embedding,
            job_embedding
        )
        
        # Calculate skills match - use extracted skills if available
        if job_skills:
            # Direct list-to-list comparison for better accuracy
            skills_score, matched_skills, missing_skills = self._calculate_skills_overlap_lists(
                resume_skills,
                job_skills
            )
        else:
            # Fallback to text-based matching
            skills_score, matched_skills, missing_skills = self.skills_matcher.calculate_skills_overlap(
                resume_skills,
                job_text
            )
        
        # Calculate experience match - use template data if available
        if job_years_required is not None:
            # Direct comparison with template data
            if resume_years and job_years_required:
                # Score based on how well experience matches requirement
                if resume_years >= job_years_required:
                    # Candidate meets or exceeds requirement
                    experience_score = min(1.0, 0.8 + (0.2 * (1 - abs(resume_years - job_years_required) / 10)))
                else:
                    # Candidate has less experience than required
                    experience_score = max(0, 0.6 * (resume_years / job_years_required))
            else:
                experience_score = 0.5  # Default if we can't compare
            logging.info(f"Experience match using template: resume={resume_years}, required={job_years_required}, score={experience_score:.2f}")
        else:
            # Fallback to text-based extraction
            experience_score = self.skills_matcher.calculate_experience_match(
                resume_years,
                job_text
            )
        
        # Calculate location/remote match
        location_score = 0.8  # Default score if location not critical
        if job_remote_policy or job_location:
            # Check resume preferences
            resume_location = resume_template.get("candidate_profile", {}).get("location", "")
            resume_remote_pref = resume_template.get("preferences", {}).get("remote_preference", "")
            
            # Score based on compatibility
            if "remote" in str(job_remote_policy).lower() or "remote" in str(job_location).lower():
                location_score = 1.0  # Remote jobs are universally compatible
            elif resume_remote_pref and "remote" in resume_remote_pref.lower():
                location_score = 0.5  # Candidate prefers remote but job might not be
            elif job_location and resume_location:
                # Check if locations match (simplified - could use geocoding)
                if any(loc in resume_location.lower() for loc in job_location.lower().split()):
                    location_score = 1.0
                else:
                    location_score = 0.6
            
            logging.info(f"Location match: job={job_location}/{job_remote_policy}, resume={resume_location}/{resume_remote_pref}, score={location_score:.2f}")
        
        # LLM assessment (if enabled)
        llm_score = 0
        llm_assessment = None
        if use_llm:
            assessment = self.llm_assessor.assess_match(
                resume_text[:500],
                job_text[:500],
                quick=True
            )
            llm_score = assessment.fit_score
            llm_assessment = assessment.recommendation
        
        # Calculate overall score with enhanced weighting
        # Adjust weights to consider location and give more weight to skills when template is available
        if job_template:
            # When we have structured data, rely more on skills and experience
            overall_score = (
                0.25 * semantic_score +  # Reduced from 0.4
                0.35 * skills_score +     # Increased from 0.3
                0.25 * experience_score + # Increased from 0.2
                0.10 * location_score +   # New factor
                0.05 * llm_score         # Reduced from 0.1
            )
            logging.info(f"Template-based scoring: semantic={semantic_score:.3f}*0.25 + skills={skills_score:.3f}*0.35 + exp={experience_score:.3f}*0.25 + loc={location_score:.3f}*0.10 + llm={llm_score:.3f}*0.05 = {overall_score:.3f}")
        else:
            # Original weights when no template
            overall_score = (
                self.config.WEIGHT_SEMANTIC * semantic_score +
                self.config.WEIGHT_SKILLS * skills_score +
                self.config.WEIGHT_EXPERIENCE * experience_score +
                self.config.WEIGHT_LLM * llm_score
            )
            logging.info(f"Text-based scoring: semantic={semantic_score:.3f}*{self.config.WEIGHT_SEMANTIC} + skills={skills_score:.3f}*{self.config.WEIGHT_SKILLS} + exp={experience_score:.3f}*{self.config.WEIGHT_EXPERIENCE} + llm={llm_score:.3f}*{self.config.WEIGHT_LLM} = {overall_score:.3f}")
        
        # Create match reasons
        match_reasons = []
        if semantic_score > 0.6:
            match_reasons.append(f"High semantic similarity ({semantic_score:.2f})")
        if skills_score > 0.5:
            match_reasons.append(f"{len(matched_skills)} matching skills")
        if experience_score > 0.7:
            if job_years_required:
                match_reasons.append(f"Experience matches ({resume_years}y vs {job_years_required}y needed)")
            else:
                match_reasons.append("Experience level matches")
        if location_score >= 1.0:
            match_reasons.append("Location/remote preference matches perfectly")
        elif location_score > 0.7:
            match_reasons.append("Location compatible")
        if llm_score > 0.7:
            match_reasons.append("Strong AI assessment")
        if job_template:
            match_reasons.append("Using structured template data")
        
        return JobMatch(
            job_id=self._get_job_id(job),
            job_title=job.get('title') or self._extract_job_title(job_text),
            company=job.get('company') or self._extract_company(job_text),
            semantic_score=semantic_score,
            skills_score=skills_score,
            experience_score=experience_score,
            llm_score=llm_score,
            overall_score=overall_score,
            match_reasons=match_reasons,
            missing_skills=missing_skills,
            llm_assessment=llm_assessment
        )
    
    def _calculate_skills_overlap_lists(
        self,
        resume_skills: List[str],
        job_skills: List[str]
    ) -> Tuple[float, List[str], List[str]]:
        """Calculate skills overlap between two lists"""
        
        # Normalize skills for comparison
        resume_skills_normalized = {s.lower().strip(): s for s in resume_skills if s}
        job_skills_normalized = {s.lower().strip(): s for s in job_skills if s}
        
        # Find matches - using sets to avoid duplicates
        matched_skills_set = set()
        matched_original = []
        
        for resume_norm, resume_orig in resume_skills_normalized.items():
            for job_norm, job_orig in job_skills_normalized.items():
                # Check for exact match or substring match
                if resume_norm == job_norm or resume_norm in job_norm or job_norm in resume_norm:
                    if resume_norm not in matched_skills_set:
                        matched_skills_set.add(resume_norm)
                        matched_original.append(resume_orig)
                    break
        
        # Find missing skills - only report skills that are in job but not in resume
        missing_skills = []
        for job_norm, job_orig in job_skills_normalized.items():
            found = False
            for resume_norm in resume_skills_normalized.keys():
                if resume_norm == job_norm or resume_norm in job_norm or job_norm in resume_norm:
                    found = True
                    break
            if not found:
                missing_skills.append(job_orig)
        
        # Calculate score
        if len(job_skills) > 0:
            skills_score = len(matched_skills_set) / len(job_skills)
        else:
            skills_score = 0.0
        
        logging.info(f"Skills match: {len(matched_skills_set)}/{len(job_skills)} = {skills_score:.2f}")
        logging.info(f"Matched skills: {matched_original[:5]}")  # Show first 5
        logging.info(f"Missing skills from job requirements: {missing_skills[:5]}")  # Show what's actually missing
        
        return skills_score, matched_original, missing_skills
    
    def _add_llm_assessment(
        self,
        match: JobMatch,
        resume_template: Dict[str, Any],
        job: Dict[str, Any]
    ) -> JobMatch:
        """Add LLM assessment to existing match"""
        
        resume_text = self._create_resume_text(resume_template)
        job_text = job.get("text", job.get("full_text", ""))
        
        assessment = self.llm_assessor.assess_match(
            resume_text[:500],
            job_text[:500],
            quick=True
        )
        
        match.llm_score = assessment.fit_score
        match.llm_assessment = assessment.recommendation
        
        # Recalculate overall score
        match.overall_score = (
            self.config.WEIGHT_SEMANTIC * match.semantic_score +
            self.config.WEIGHT_SKILLS * match.skills_score +
            self.config.WEIGHT_EXPERIENCE * match.experience_score +
            self.config.WEIGHT_LLM * match.llm_score
        )
        
        return match
    
    def _create_resume_text(self, resume_template: Dict[str, Any]) -> str:
        """Create searchable text from resume template"""
        
        parts = []
        
        # Add summary
        if "qualifications" in resume_template:
            parts.append(resume_template["qualifications"].get("experience_summary", ""))
        
        # Add skills
        if "requirements_match" in resume_template:
            skills = resume_template["requirements_match"].get("technical_skills", [])
            # Handle skills that might be dicts
            skill_strings = []
            for skill in skills:
                if isinstance(skill, dict):
                    skill_name = skill.get('skill_name', skill.get('name', skill.get('skill', '')))
                    if skill_name:
                        skill_strings.append(skill_name)
                elif isinstance(skill, str):
                    skill_strings.append(skill)
            if skill_strings:
                parts.append(" ".join(skill_strings))
        
        # Add work history
        if "work_history" in resume_template:
            for job in resume_template["work_history"]:
                parts.append(f"{job.get('title', '')} {job.get('company', '')}")
        
        return " ".join(parts)
    
    def _get_job_id(self, job: Dict[str, Any]) -> str:
        """Extract or generate job ID"""
        # Check for job_id first (from database), then id, then job_index
        return job.get("job_id", job.get("id", job.get("job_index", str(hash(str(job))))))
    
    def _extract_job_title(self, job_text: str) -> Optional[str]:
        """Extract job title from text"""
        import re
        
        # Handle concatenated text (when everything is on one line)
        if '\n' not in job_text or job_text.count('\n') < 3:
            # Try to find job title patterns in concatenated text
            # Look for patterns like "Company Name Job Title Company Name • Location" 
            
            # Common job title keywords (expanded list)
            title_keywords = [
                'specialist', 'manager', 'engineer', 'developer', 'analyst',
                'coordinator', 'director', 'associate', 'consultant', 'administrator',
                'technician', 'designer', 'architect', 'lead', 'senior', 'junior',
                'supervisor', 'executive', 'officer', 'assistant', 'accountant',
                'representative', 'agent', 'advisor', 'expert', 'professional',
                'strategist', 'scientist', 'researcher', 'trainer', 'instructor',
                'payroll', 'billing', 'fractional'  # Add specific keywords from our examples
            ]
            
            # New approach: Handle pattern where company name appears twice
            # e.g., "Confidential Careers Senior Payroll Manager - Fully Remote ( Dallas Homebase Required) Confidential Careers • Dallas, TX"
            first_200_chars = job_text[:200] if len(job_text) > 200 else job_text
            
            # Try to find repeated company name pattern
            words = first_200_chars.split()
            if len(words) >= 4:
                # Skip single letter prefixes (like "E" in "E Everything To Gain")
                start_idx = 0
                if len(words[0]) == 1 and len(words) > 1:
                    start_idx = 1
                    words = words[1:]  # Skip the single letter
                
                # Check for pattern: [Company] [Title] [Company] • [Location]
                for company_word_count in [3, 2, 1]:  # Try different company name lengths
                    if len(words) > company_word_count + 2:
                        potential_company = ' '.join(words[:company_word_count])
                        remaining_text = ' '.join(words[company_word_count:])
                        
                        # Check if company name appears again (case-insensitive)
                        if potential_company.lower() in remaining_text.lower():
                            # Extract text between the two company names
                            # Find where company name appears again
                            company_second_pos = remaining_text.lower().find(potential_company.lower())
                            if company_second_pos > 0:
                                title_section = remaining_text[:company_second_pos].strip()
                                
                                # Clean up the extracted title
                                # Remove location/remote indicators
                                title_section = re.sub(r'\s*-\s*Fully\s+Remote.*', '', title_section, flags=re.IGNORECASE)
                                title_section = re.sub(r'\s*-\s*Remote.*', '', title_section, flags=re.IGNORECASE)
                                title_section = re.sub(r'\s*-\s*US\s*$', '', title_section, flags=re.IGNORECASE)
                                title_section = re.sub(r'\s*\([^)]*\)\s*$', '', title_section)
                                title_section = title_section.strip(' -')
                                
                                # Validate it's a real title (contains keywords and reasonable length)
                                if title_section and 2 <= len(title_section.split()) <= 8:
                                    title_lower = title_section.lower()
                                    if any(keyword in title_lower for keyword in title_keywords):
                                        return title_section.strip()
            
            # Fallback: Split by common separators
            parts = re.split(r'[•·|]', job_text)
            if len(parts) > 1:
                # First part often contains company and title
                first_part = parts[0].strip()
                
                # Try to extract title from first part
                words = first_part.split()
                for i, word in enumerate(words):
                    if word.lower() in title_keywords:
                        # Found a title keyword, extract surrounding words
                        start = max(0, i - 2)
                        end = min(len(words), i + 3)
                        potential_title = ' '.join(words[start:end])
                        
                        # Clean up company names that might be prefixed
                        for company_indicator in ['Group', 'Inc', 'LLC', 'Corp', 'Company', 'Companies']:
                            if company_indicator in potential_title:
                                # Split at company name and take the part after
                                parts = potential_title.split(company_indicator)
                                if len(parts) > 1 and parts[1].strip():
                                    potential_title = parts[1].strip()
                                    break
                        
                        # Remove location indicators
                        for loc in [' - Remote', ' - ', 'Remote', '(Remote)']:
                            potential_title = potential_title.replace(loc, '').strip()
                        
                        if potential_title and len(potential_title.split()) <= 6:
                            return potential_title
            
            # Try regex patterns for common title formats
            patterns = [
                # Sr./Jr. Accountant, Payroll & General Accounting (Remote)
                r'((?:Sr\.|Jr\.|Senior|Junior)\s+[A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*(?:\s+&\s+[A-Z][a-z]+\s+[A-Z][a-z]+)?)\s*\([Rr]emote\)',
                # Title with keywords followed by separator
                r'([A-Z][a-z]+ (?:' + '|'.join(title_keywords) + r')[^•·|()]*?)(?:\s*[-–]\s*Remote|\s*•|\s*\||\s*\()',
                # Sr./Junior/Senior prefix with any title
                r'((?:Sr\.|Jr\.|Senior|Junior)\s+[A-Z][a-z]+(?:,?\s+[A-Z][a-z]+)*)',
                # Generic title pattern with keywords
                r'((?:Sr\.|Senior|Junior|Jr\.)?\s*[A-Z][a-z]+\s+(?:' + '|'.join(title_keywords) + r')[^•·|]*)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, job_text, re.IGNORECASE)
                if match:
                    title = match.group(1).strip()
                    # Clean up
                    title = re.sub(r'\s+', ' ', title)
                    if title and len(title.split()) <= 6:
                        return title
        
        # Original logic for multi-line text
        lines = job_text.split('\n')
        
        # Look for common job title patterns
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if not line:
                continue
            
            # Skip common non-title lines
            if any(skip in line.lower() for skip in ['apply', 'posted', 'location', 'salary', 'http', 'www']):
                continue
            
            # Look for title patterns
            # Titles often have capitalized words and are 2-8 words long
            words = line.split()
            if 2 <= len(words) <= 8:
                # Check if it looks like a title (has some capitalized words)
                cap_words = sum(1 for w in words if w and w[0].isupper())
                if cap_words >= len(words) * 0.4:  # At least 40% capitalized
                    return line
        
        # Fallback: just return first non-empty line
        for line in lines[:5]:
            line = line.strip()
            if line and len(line) > 5 and len(line) < 100:
                return line
        
        return "Unknown Position"
    
    def _extract_company(self, job_text: str) -> Optional[str]:
        """Extract company from text"""
        import re
        
        # Handle concatenated text first
        if '\n' not in job_text or job_text.count('\n') < 3:
            # Split by common separators
            parts = re.split(r'[•·|]', job_text)
            
            if len(parts) > 0:
                # First part often starts with company name
                first_part = parts[0].strip()
                
                # Look for company name patterns
                # Skip single letters at the beginning (like "U" for UnitedHealth)
                text_to_check = first_part
                if len(text_to_check) > 2 and text_to_check[1] == ' ':
                    text_to_check = text_to_check[2:].strip()
                
                # Common company name patterns
                company_patterns = [
                    r'^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s+(?:Group|Inc|LLC|Corp|Company|Companies)',
                    r'^((?:HFW|IBM|AWS|GE|HP|3M|[A-Z]{2,4})\s+(?:Group|Inc|LLC|Corp|Company|Companies))',
                    r'^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2})\s+(?:Payroll|Sr\.|Senior|Junior)',
                ]
                
                for pattern in company_patterns:
                    match = re.match(pattern, text_to_check)
                    if match:
                        company = match.group(1).strip()
                        # Add back common suffixes if they were part of the name
                        if 'Group' in text_to_check and 'Group' not in company:
                            company += ' Group'
                        elif 'Companies' in text_to_check and 'Companies' not in company:
                            company += ' Companies'
                        
                        # Limit company name length (probably extracted too much)
                        if len(company) > 50:
                            # Try to cut at a reasonable point
                            words = company.split()
                            if len(words) > 4:
                                company = ' '.join(words[:3])
                        
                        return company
            
            # Try to find company name before bullet point
            match = re.search(r'^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)*)\s*[•·|]', job_text)
            if match:
                return match.group(1).strip()
        
        # Original logic for multi-line text
        lines = job_text.split('\n')
        
        # Look for company patterns
        for i, line in enumerate(lines[:20]):
            line = line.strip()
            
            # Look for "at Company" or "Company Name" patterns
            if ' at ' in line.lower():
                parts = line.lower().split(' at ')
                if len(parts) > 1:
                    company = parts[1].strip()
                    # Clean up common suffixes
                    for suffix in [',', '.', ' -', ' |']:
                        if suffix in company:
                            company = company.split(suffix)[0]
                    return company.title()
            
            # Look for lines after job title that might be company
            if i > 0 and i < 10:
                # Check if previous line was likely a title
                prev_line = lines[i-1].strip()
                if prev_line and len(prev_line.split()) <= 8:
                    # This line might be company if it's short and doesn't look like description
                    if len(line.split()) <= 5 and not any(skip in line.lower() for skip in ['job', 'position', 'we are', 'looking', 'salary']):
                        return line
        
        return None


# -------------------------
# LangGraph Node Interface
# -------------------------

def job_matching_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible node for job matching.
    
    Expected state keys:
        - resume_templates: List of resume templates to match
        - jobs: List of job postings to match against
        - use_llm: Whether to use LLM assessment (default True)
        - config (optional): Configuration overrides
        - verbose (optional): Enable verbose logging
    
    Returns state with:
        - matching_results: List of MatchingResult objects
        - best_matches: Top matches across all resumes
        - diagnostics: Processing diagnostics
    """
    
    # Set up logging
    if state.get("verbose", False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    # Get inputs
    resume_templates = state.get("resume_templates", [])
    jobs = state.get("jobs", [])
    
    if not resume_templates or not jobs:
        state["matching_results"] = []
        state["best_matches"] = []
        state["diagnostics"] = {
            "job_matching": {
                "status": "no_data",
                "resumes_count": len(resume_templates),
                "jobs_count": len(jobs)
            }
        }
        return state
    
    # Load configuration
    config = JobMatchConfig()
    if "config" in state and isinstance(state["config"], dict):
        config = JobMatchConfig.from_dict(state["config"])
    
    # Initialize pipeline
    pipeline = JobMatchingPipeline(config=config)
    
    try:
        # Process each resume
        all_results = []
        all_matches = []
        
        for resume_template in resume_templates:
            logging.info(f"Processing resume: {resume_template.get('candidate_profile', {}).get('name', 'Unknown')}")
            
            result = pipeline.match_resume_to_jobs(
                resume_template,
                jobs,
                use_llm=state.get("use_llm", True),
                verbose=state.get("verbose", False)
            )
            
            all_results.append(result)
            
            # Collect all matches with resume info
            for match in result.matches:
                all_matches.append({
                    "resume_name": result.candidate_name,
                    "resume_id": result.resume_id,
                    "job_id": match.job_id,
                    "job_title": match.job_title,
                    "company": match.company,
                    "overall_score": match.overall_score,
                    "match_reasons": match.match_reasons,
                    "llm_assessment": match.llm_assessment
                })
        
        # Sort all matches by score
        all_matches.sort(key=lambda x: x["overall_score"], reverse=True)
        
        # Convert results to dictionaries
        state["matching_results"] = [
            {
                "resume_id": r.resume_id,
                "candidate_name": r.candidate_name,
                "total_jobs_evaluated": r.total_jobs_evaluated,
                "matches": [
                    {
                        "job_id": m.job_id,
                        "job_title": m.job_title,
                        "company": m.company,
                        "overall_score": m.overall_score,
                        "semantic_score": m.semantic_score,
                        "skills_score": m.skills_score,
                        "experience_score": m.experience_score,
                        "llm_score": m.llm_score,
                        "match_reasons": m.match_reasons,
                        "missing_skills": m.missing_skills,
                        "llm_assessment": m.llm_assessment
                    } for m in r.matches
                ],
                "processing_time": r.processing_time,
                "timestamp": r.timestamp
            } for r in all_results
        ]
        
        state["best_matches"] = all_matches[:20]  # Top 20 overall
        
        if "diagnostics" not in state:
            state["diagnostics"] = {}
        state["diagnostics"]["job_matching"] = {
            "status": "success",
            "resumes_processed": len(all_results),
            "jobs_evaluated": len(jobs),
            "total_matches_found": len(all_matches),
            "average_processing_time": sum(r.processing_time for r in all_results) / len(all_results)
        }
        
    except Exception as e:
        logging.error(f"Job matching failed: {e}")
        state["matching_results"] = []
        state["best_matches"] = []
        state["diagnostics"] = {
            "job_matching": {
                "status": "error",
                "error": str(e)
            }
        }
    
    return state


# -------------------------
# Standalone Testing
# -------------------------

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print("JOB MATCHING TEST")
    print("=" * 60)
    
    # Load test data
    resume_file = "data/processed/resume_template_sample.json"
    jobs_file = "data/jobs/bulk_latest.json"
    
    if not Path(resume_file).exists():
        print(f"Resume template not found: {resume_file}")
        print("Run test_resume_processing.py first")
        sys.exit(1)
    
    if not Path(jobs_file).exists():
        print(f"Jobs file not found: {jobs_file}")
        print("Run the job crawler first")
        sys.exit(1)
    
    # Load data
    with open(resume_file, 'r') as f:
        resume_template = json.load(f)
    
    with open(jobs_file, 'r') as f:
        jobs_data = json.load(f)
        jobs = jobs_data.get("jobs", [])
    
    print(f"\nLoaded resume template")
    print(f"Loaded {len(jobs)} jobs")
    
    # Test matching
    print("\n" + "-" * 40)
    print("TESTING JOB MATCHING")
    print("-" * 40)
    
    pipeline = JobMatchingPipeline()
    
    result = pipeline.match_resume_to_jobs(
        resume_template,
        jobs,
        use_llm=True,
        verbose=True
    )
    
    print(f"\n" + "=" * 40)
    print("MATCHING RESULTS")
    print("=" * 40)
    
    print(f"Candidate: {result.candidate_name}")
    print(f"Jobs evaluated: {result.total_jobs_evaluated}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Matches found: {len(result.matches)}")
    
    print("\nTOP MATCHES:")
    for i, match in enumerate(result.matches[:5], 1):
        print(f"\n{i}. Job ID: {match.job_id}")
        print(f"   Title: {match.job_title or 'Unknown'}")
        print(f"   Overall Score: {match.overall_score:.3f}")
        print(f"   - Semantic: {match.semantic_score:.3f}")
        print(f"   - Skills: {match.skills_score:.3f}")
        print(f"   - Experience: {match.experience_score:.3f}")
        print(f"   - LLM: {match.llm_score:.3f}")
        if match.match_reasons:
            print(f"   Reasons: {', '.join(match.match_reasons)}")
        if match.llm_assessment:
            print(f"   LLM says: {match.llm_assessment[:100]}...")
    
    # Save results
    output_file = "data/processed/job_matches_latest.json"
    output_data = {
        "candidate_name": result.candidate_name,
        "timestamp": result.timestamp,
        "jobs_evaluated": result.total_jobs_evaluated,
        "matches": [
            {
                "job_id": m.job_id,
                "job_title": m.job_title,
                "scores": {
                    "overall": m.overall_score,
                    "semantic": m.semantic_score,
                    "skills": m.skills_score,
                    "experience": m.experience_score,
                    "llm": m.llm_score
                },
                "assessment": m.llm_assessment
            } for m in result.matches
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

"""
LanceDB integration for job matching system
Provides vector storage with section-level embeddings for improved matching
"""

import os
import json
import logging
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from pydantic import BaseModel, Field
from typing import Annotated

from graph.embeddings import JobEmbeddings
from graph.settings import get_settings

logger = logging.getLogger(__name__)


# -------------------------
# Data Models
# -------------------------

class JobPosting(LanceModel):
    """Schema for job postings with section-level embeddings"""
    # Core identifiers
    job_id: str
    
    # Raw text content
    text: str  # Full job text from crawler
    text_length: int
    preview: str
    
    # Extracted structured data
    title: Optional[str] = None
    company: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    employment_type: Optional[str] = None  # Full-time, Part-time, Contract, etc.
    remote_policy: Optional[str] = None  # Remote, Hybrid, Onsite
    
    # Extracted sections (JSON strings)
    description: Optional[str] = None  # Job description section
    requirements: Optional[str] = None  # Requirements/qualifications section as JSON array
    responsibilities: Optional[str] = None  # Responsibilities section as JSON array
    benefits: Optional[str] = None  # Benefits section as JSON array
    skills: Optional[str] = None  # Extracted skills as JSON array
    
    # Metadata
    job_index: int
    search_query: str
    search_location: str
    crawl_timestamp: str
    posted_date: Optional[str] = None  # When the job was posted (e.g., "2 days ago")
    employment_type: Optional[str] = None  # Full-time, Part-time, Contract, etc.
    via: Optional[str] = None  # Source of the job posting (via Indeed, via LinkedIn, etc.)
    apply_links: Optional[str] = None  # JSON array of apply links (url and text)
    
    # Processed template (JSON string)
    vocation_template: Optional[str] = None
    
    # Vector embeddings for different sections (all required, but can be zero vectors)
    embedding_full: Annotated[list[float], Vector(768)]  # Full text embedding
    embedding_title: Annotated[list[float], Vector(768)]  # Title embedding
    embedding_description: Annotated[list[float], Vector(768)]  # Description embedding
    embedding_requirements: Annotated[list[float], Vector(768)]  # Requirements embedding
    embedding_skills: Annotated[list[float], Vector(768)]  # Skills embedding


class Resume(LanceModel):
    """Schema for resumes with section-level embeddings"""
    # Identifiers
    resume_id: str
    filename: str
    
    # Contact information
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    
    # Full text
    full_text: str
    
    # Extracted sections as text
    summary: Optional[str] = None  # Professional summary
    
    # Extracted structured data (JSON strings)
    experience: Optional[str] = None  # Work experience as JSON array
    education: Optional[str] = None   # Education as JSON array
    skills: Optional[str] = None      # Skills as JSON array with categories
    certifications: Optional[str] = None  # Certifications as JSON array
    achievements: Optional[str] = None  # Achievements as JSON array
    
    # Computed fields
    years_experience: Optional[float] = None
    
    # Preferences (JSON strings)
    salary_expectations: Optional[str] = None  # JSON with min/max
    work_preferences: Optional[str] = None  # Remote, location, etc.
    
    # Processed templates for matching
    matching_template: Optional[str] = None  # JSON string
    vocation_template: Optional[str] = None  # JSON string - job seeker preferences
    
    # Metadata
    process_timestamp: str
    
    # Vector embeddings for different sections (all required, but can be zero vectors)
    embedding_full: Annotated[list[float], Vector(768)]  # Full text embedding
    embedding_summary: Annotated[list[float], Vector(768)]  # Summary embedding
    embedding_experience: Annotated[list[float], Vector(768)]  # Experience embedding
    embedding_skills: Annotated[list[float], Vector(768)]  # Skills embedding
    embedding_combined: Annotated[list[float], Vector(768)]  # Combined key sections


class AnalyticsReport(LanceModel):
    """Schema for analytics reports tied to specific resumes"""
    # Primary key is resume_id (one report per resume)
    resume_id: str  # Primary key and foreign key to Resume table
    
    # Report metadata
    report_timestamp: str
    generation_date: str
    
    # Report statistics
    total_jobs_analyzed: int
    total_matches_analyzed: int
    avg_match_score: float
    match_distribution: str  # JSON string with excellent/good/fair/poor counts
    
    # Skills analysis
    top_demanded_skills: str  # JSON array of {skill, count} objects
    skill_gaps: str  # JSON array of missing skills
    skill_strengths: str  # JSON array of aligned skills
    
    # Market analysis
    top_job_titles: str  # JSON array of {title, count} objects
    
    # LLM-generated insights
    market_insights: str  # Full text of market insights
    recommendations: str  # Full text of recommendations
    
    # User preferences at time of report
    user_preferences: str  # JSON string of preferences used
    
    # Full report JSON for future reference
    full_report_json: str  # Complete report data as JSON


class JobMatch(LanceModel):
    """Schema for job-resume matches with detailed scoring"""
    # Identifiers
    match_id: str
    job_id: str
    resume_id: str
    
    # Overall scores
    overall_score: float
    
    # Component scores
    semantic_score: float  # Full text similarity
    title_match_score: float  # How well resume matches job title
    skills_score: float  # Skills overlap
    experience_score: float  # Experience level match
    requirements_score: float  # Requirements match
    education_score: Optional[float] = None  # Education match
    location_score: Optional[float] = None  # Location preference match
    salary_score: Optional[float] = None  # Salary alignment
    
    # Section-level semantic scores
    summary_to_description_score: Optional[float] = None
    experience_to_requirements_score: Optional[float] = None
    skills_to_skills_score: Optional[float] = None
    
    # Preference alignment
    location_preference_met: Optional[bool] = None
    remote_preference_met: Optional[bool] = None
    salary_match: Optional[str] = None  # JSON object with salary details
    
    # LLM analysis
    llm_score: Optional[float] = None
    llm_assessment: Optional[str] = None
    
    # Match analysis (JSON strings)
    match_reasons: str  # JSON array of reasons
    skills_matched: Optional[str] = None  # JSON array of matched skills
    skills_gap: Optional[str] = None  # JSON array of missing skills
    requirements_matched: Optional[str] = None  # JSON array
    requirements_gap: Optional[str] = None  # JSON array
    
    # Metadata
    match_timestamp: str
    pipeline_run_id: Optional[str] = None


# -------------------------
# Database Manager
# -------------------------

class graphDB:
    """LanceDB manager for job matching system with section-level embeddings"""
    
    def __init__(self, db_path: Optional[str] = None, reset_if_exists: Optional[bool] = None):
        """Initialize database connection
        
        Args:
            db_path: Path to database directory (uses settings if not provided)
            reset_if_exists: If True, delete and recreate database (uses settings if not provided)
        """
        # Get settings
        settings = get_settings()
        
        # Use provided values or fall back to settings
        self.db_path = db_path or settings.database.db_path
        reset = reset_if_exists if reset_if_exists is not None else settings.database.reset_if_exists
        
        # Store settings for later use
        self.settings = settings
        
        # Optionally reset database
        if reset and os.path.exists(self.db_path):
            logger.info(f"Resetting database at {self.db_path}")
            shutil.rmtree(self.db_path)
        
        self.db = lancedb.connect(self.db_path)
        self.embedder = JobEmbeddings()
        
        # Initialize tables
        self._init_tables()
        
    def _init_tables(self):
        """Create tables if they don't exist"""
        table_schemas = {
            "jobs": JobPosting,
            "resumes": Resume,
            "matches": JobMatch
        }
        
        existing_tables = self.db.table_names()
        
        for table_name, schema in table_schemas.items():
            if table_name not in existing_tables:
                logger.info(f"Creating table: {table_name}")
                # Create empty table with schema
                self.db.create_table(table_name, schema=schema.to_arrow_schema())
    
    def drop_table(self, table_name: str):
        """Drop a table if it exists"""
        if table_name in self.db.table_names():
            self.db.drop_table(table_name)
            logger.info(f"Dropped table: {table_name}")
    
    def drop_all_tables(self):
        """Drop all tables"""
        for table_name in self.db.table_names():
            self.drop_table(table_name)
    
    def recreate_table(self, table_name: str):
        """Recreate a table with fresh schema"""
        self.drop_table(table_name)
        self._init_tables()
    
    def recreate_all_tables(self):
        """Recreate all tables with fresh schema"""
        self.drop_all_tables()
        self._init_tables()
    
    # -------------------------
    # Job Operations
    # -------------------------
    
    def add_job(self, job_data: Dict[str, Any], search_query: str = "", search_location: str = "remote") -> str:
        """Add a single processed job with section-level embeddings"""
        
        # Extract text sections for duplicate checking
        full_text = job_data.get('text', '')
        title = job_data.get('title', '')
        company = job_data.get('company', '')
        
        # Check for duplicate job based on title + company combination
        # This is more reliable than checking full text which might have minor variations
        try:
            table = self.db.open_table("jobs")
            existing_df = table.to_pandas()
            
            if not existing_df.empty and 'title' in existing_df.columns and 'company' in existing_df.columns:
                # Check if this exact job already exists
                duplicate_exists = ((existing_df['title'] == title) & 
                                  (existing_df['company'] == company)).any()
                
                if duplicate_exists:
                    # Find the existing job's ID and return it
                    existing_job = existing_df[(existing_df['title'] == title) & 
                                              (existing_df['company'] == company)].iloc[0]
                    existing_job_id = existing_job['job_id']
                    logger.info(f"Job already exists: '{title}' at '{company}' (ID: {existing_job_id})")
                    return existing_job_id
                    
        except Exception as e:
            # If table doesn't exist yet, continue with adding the job
            logger.debug(f"Could not check for duplicates: {e}")
        
        timestamp = datetime.now().isoformat()
        job_id = f"job_{timestamp}_{job_data.get('job_index', 0)}"
        
        # Continue with rest of the processing
        description = job_data.get('description', '')
        requirements = job_data.get('requirements', [])
        skills = job_data.get('skills', [])
        
        # Generate embeddings for each section (use zero vector if section missing)
        zero_vector = [0.0] * 768  # Zero vector for missing sections
        
        embeddings = {}
        embeddings['embedding_full'] = self.embedder.encode([full_text[:3000]])[0].tolist()
        
        if title:
            embeddings['embedding_title'] = self.embedder.encode([title])[0].tolist()
        else:
            embeddings['embedding_title'] = zero_vector
            
        if description:
            embeddings['embedding_description'] = self.embedder.encode([description[:1500]])[0].tolist()
        else:
            embeddings['embedding_description'] = zero_vector
            
        if requirements:
            if isinstance(requirements, list):
                # Convert all items to strings before joining
                req_text = ' '.join(str(item) for item in requirements)
            else:
                req_text = str(requirements)
            embeddings['embedding_requirements'] = self.embedder.encode([req_text[:1500]])[0].tolist()
        else:
            embeddings['embedding_requirements'] = zero_vector
            
        if skills:
            if isinstance(skills, list):
                # Convert all items to strings before joining
                skills_text = ' '.join(str(skill) for skill in skills)
            else:
                skills_text = str(skills)
            embeddings['embedding_skills'] = self.embedder.encode([skills_text])[0].tolist()
        else:
            embeddings['embedding_skills'] = zero_vector
        
        # Create record
        record = {
            "job_id": job_id,
            "text": full_text,
            "text_length": len(full_text),
            "preview": full_text[:200],
            "title": title,
            "company": job_data.get('company'),
            "location": job_data.get('location', search_location),
            "salary_min": job_data.get('salary_min'),
            "salary_max": job_data.get('salary_max'),
            "employment_type": job_data.get('employment_type'),
            "remote_policy": job_data.get('remote_policy'),
            "description": description,
            "requirements": json.dumps(requirements) if requirements else None,
            "responsibilities": json.dumps(job_data.get('responsibilities', [])) if job_data.get('responsibilities') else None,
            "benefits": json.dumps(job_data.get('benefits', [])) if job_data.get('benefits') else None,
            "skills": json.dumps(skills) if skills else None,
            "job_index": job_data.get('job_index', 0),
            "search_query": search_query,
            "search_location": search_location,
            "crawl_timestamp": timestamp,
            "posted_date": job_data.get('posted_date'),  # Add posted date from crawler
            "employment_type": job_data.get('employment_type'),  # Add employment type
            "via": job_data.get('via'),  # Add source
            "apply_links": json.dumps(job_data.get('apply_links', [])) if job_data.get('apply_links') else None,  # Add apply links
            "vocation_template": json.dumps(job_data.get('vocation_template', {})) if job_data.get('vocation_template') else None,
            **embeddings
        }
        
        # Add to table
        table = self.db.open_table("jobs")
        table.add([record])
        logger.info(f"Added job {job_id} with section embeddings")
        
        return job_id
    
    def add_jobs_batch(self, jobs: List[Dict[str, Any]], search_query: str, search_location: str = "remote") -> Tuple[int, List[str]]:
        """Add multiple jobs from crawler output, with duplicate checking
        
        Returns:
            Tuple of (count of added jobs, list of job IDs that were added/updated)
        """
        
        # Get existing job texts for duplicate checking
        existing_jobs = set()
        try:
            table = self.db.open_table("jobs")
            existing_df = table.to_pandas()
            if not existing_df.empty and 'text' in existing_df.columns:
                existing_jobs = set(existing_df['text'].apply(lambda x: x[:500] if x else '').tolist())
                logger.info(f"Found {len(existing_jobs)} existing jobs in database")
        except Exception as e:
            logger.warning(f"Could not load existing jobs for duplicate check: {e}")
            table = self.db.open_table("jobs")
        
        added = 0
        skipped = 0
        updated_job_ids = []
        
        for job in jobs:
            job_text = job.get('text', '')
            if not job_text:
                continue
            
            try:
                job_id = self.add_job(job, search_query, search_location)
                added += 1
                updated_job_ids.append(job_id)
            except Exception as e:
                logger.error(f"Error adding job: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        logger.info(f"Added {added} jobs to database")
        return added, updated_job_ids
    
    # -------------------------
    # Resume Operations
    # -------------------------
    
    def add_resume(self, resume_data: Dict[str, Any]) -> str:
        """Add a processed resume with section-level embeddings"""
        
        timestamp = datetime.now().isoformat()
        
        # Debug logging
        logger.info(f"add_resume called with data keys: {resume_data.keys()}")
        
        # Generate unique resume ID
        contact_info = resume_data.get('contact_info', {})
        if contact_info is None:
            contact_info = {}
        email = contact_info.get('email', '') if isinstance(contact_info, dict) else ''
        name = contact_info.get('name', 'unknown') if isinstance(contact_info, dict) else 'unknown'
        resume_id = f"resume_{email}_{timestamp}" if email else f"resume_{name}_{timestamp}"
        
        # Extract text sections
        full_text = resume_data.get('full_text', '')
        summary = resume_data.get('professional_summary', '')
        experience = resume_data.get('work_experience', [])
        skills = resume_data.get('technical_skills', []) + resume_data.get('frameworks_tools', [])
        
        # Generate embeddings for each section (use zero vector if section missing)
        zero_vector = [0.0] * 768  # Zero vector for missing sections
        
        embeddings = {}
        embeddings['embedding_full'] = self.embedder.encode([full_text[:3000]])[0].tolist()
        
        if summary:
            embeddings['embedding_summary'] = self.embedder.encode([summary])[0].tolist()
        else:
            embeddings['embedding_summary'] = zero_vector
            
        if experience:
            # Create experience text from structured data
            exp_texts = []
            for exp in experience:
                exp_text = f"{exp.get('title', '')} at {exp.get('company', '')}. "
                if exp.get('responsibilities'):
                    exp_text += ' '.join(exp['responsibilities'][:3])
                exp_texts.append(exp_text)
            experience_text = ' '.join(exp_texts[:3])  # Use first 3 experiences
            embeddings['embedding_experience'] = self.embedder.encode([experience_text[:1500]])[0].tolist()
        else:
            embeddings['embedding_experience'] = zero_vector
            
        if skills:
            skills_text = ' '.join(skills)
            embeddings['embedding_skills'] = self.embedder.encode([skills_text])[0].tolist()
        else:
            embeddings['embedding_skills'] = zero_vector
            
        # Create combined embedding from key sections
        combined_text = f"{summary} {' '.join(skills[:10]) if skills else ''}"
        if experience and len(experience) > 0:
            combined_text += f" {experience[0].get('title', '')} {experience[0].get('company', '')}"
        if combined_text.strip():
            embeddings['embedding_combined'] = self.embedder.encode([combined_text[:1500]])[0].tolist()
        else:
            embeddings['embedding_combined'] = zero_vector
        
        # Prepare contact info
        contact = resume_data.get('contact_info', {})
        if contact is None or not isinstance(contact, dict):
            contact = {}
        
        # Create record
        record = {
            "resume_id": resume_id,
            "filename": resume_data.get('filename', 'unknown.txt'),
            "name": contact.get('name') if contact else None,
            "email": contact.get('email') if contact else None,
            "phone": contact.get('phone') if contact else None,
            "location": contact.get('location') if contact else None,
            "linkedin": contact.get('linkedin') if contact else None,
            "github": contact.get('github') if contact else None,
            "full_text": full_text,
            "summary": summary,
            "experience": json.dumps(experience) if experience else None,
            "education": json.dumps(resume_data.get('education', [])) if resume_data.get('education') else None,
            "skills": json.dumps({
                'technical': resume_data.get('technical_skills', []),
                'languages': resume_data.get('programming_languages', []),
                'tools': resume_data.get('frameworks_tools', [])
            }),
            "certifications": json.dumps(resume_data.get('certifications', [])) if resume_data.get('certifications') else None,
            "achievements": json.dumps(resume_data.get('achievements', [])) if resume_data.get('achievements') else None,
            "years_experience": resume_data.get('years_of_experience'),
            "salary_expectations": json.dumps(resume_data.get('salary_expectations', {})) if resume_data.get('salary_expectations') else None,
            "work_preferences": json.dumps(resume_data.get('location_preferences', {})) if resume_data.get('location_preferences') else None,
            "matching_template": json.dumps(resume_data.get('matching_template', {})) if resume_data.get('matching_template') else None,
            "vocation_template": json.dumps(resume_data.get('vocation_template', {})) if resume_data.get('vocation_template') else None,
            "process_timestamp": timestamp,
            **embeddings
        }
        
        # Check for duplicate by filename
        try:
            table = self.db.open_table("resumes")
            existing_df = table.to_pandas()
            if not existing_df.empty and 'filename' in existing_df.columns:
                existing_files = set(existing_df['filename'].tolist())
                if record['filename'] in existing_files:
                    logger.info(f"Updating existing resume: {record['filename']}")
                    # Delete old entry
                    # Note: LanceDB doesn't have direct update, so we'll add as new version
        except Exception as e:
            logger.warning(f"Could not check for existing resume: {e}")
            table = self.db.open_table("resumes")
        
        # Add to table
        table.add([record])
        logger.info(f"Added resume {resume_id} with section embeddings")
        
        return resume_id
    
    def search_resumes_by_job(self, job_id: str, limit: int = 10) -> List[Dict]:
        """Search for resumes that match a specific job using section embeddings"""
        
        # Get the job
        jobs_table = self.db.open_table("jobs")
        job_df = jobs_table.to_pandas()
        job = job_df[job_df['job_id'] == job_id]
        
        if job.empty:
            logger.warning(f"Job {job_id} not found")
            return []
        
        job = job.iloc[0].to_dict()
        
        # Get resumes table
        resumes_table = self.db.open_table("resumes")
        
        # Perform multiple searches with different embedding combinations
        results = []
        
        # Helper function to check if embedding is non-zero
        def is_valid_embedding(embedding):
            if embedding is None:
                return False
            # Convert to list if numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            return any(v != 0 for v in embedding[:10] if isinstance(v, (int, float)))
        
        # 1. Search by job requirements against resume experience
        if is_valid_embedding(job.get('embedding_requirements')):
            emb = job['embedding_requirements']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            req_results = resumes_table.search(
                emb,
                vector_column_name="embedding_experience"
            ).limit(limit).to_list()
            results.extend(req_results)
        
        # 2. Search by job description against resume summary
        if is_valid_embedding(job.get('embedding_description')):
            emb = job['embedding_description']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            desc_results = resumes_table.search(
                emb,
                vector_column_name="embedding_summary"
            ).limit(limit).to_list()
            results.extend(desc_results)
        
        # 3. Search by job skills against resume skills
        if is_valid_embedding(job.get('embedding_skills')):
            emb = job['embedding_skills']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            skill_results = resumes_table.search(
                emb,
                vector_column_name="embedding_skills"
            ).limit(limit).to_list()
            results.extend(skill_results)
        
        # 4. Full text search as fallback
        emb = job['embedding_full']
        if hasattr(emb, 'tolist'):
            emb = emb.tolist()
        full_results = resumes_table.search(
            emb,
            vector_column_name="embedding_full"
        ).limit(limit).to_list()
        results.extend(full_results)
        
        # Deduplicate by resume_id and return top matches
        seen = set()
        unique_results = []
        for r in results:
            if r['resume_id'] not in seen:
                seen.add(r['resume_id'])
                unique_results.append(r)
                if len(unique_results) >= limit:
                    break
        
        return unique_results
    
    def search_jobs_by_resume(self, resume_id: str, limit: int = 10) -> List[Dict]:
        """Search for jobs that match a specific resume using section embeddings"""
        
        # Get the resume
        resumes_table = self.db.open_table("resumes")
        resume_df = resumes_table.to_pandas()
        resume = resume_df[resume_df['resume_id'] == resume_id]
        
        if resume.empty:
            logger.warning(f"Resume {resume_id} not found")
            return []
        
        resume = resume.iloc[0].to_dict()
        
        # Get jobs table
        jobs_table = self.db.open_table("jobs")
        
        # Perform multiple searches with different embedding combinations
        results = []
        
        # Helper function to check if embedding is non-zero
        def is_valid_embedding(embedding):
            if embedding is None:
                return False
            # Convert to list if numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            return any(v != 0 for v in embedding[:10] if isinstance(v, (int, float)))
        
        # 1. Search by resume experience against job requirements
        if is_valid_embedding(resume.get('embedding_experience')):
            emb = resume['embedding_experience']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            exp_results = jobs_table.search(
                emb,
                vector_column_name="embedding_requirements"
            ).limit(limit).to_list()
            results.extend(exp_results)
        
        # 2. Search by resume summary against job description
        if is_valid_embedding(resume.get('embedding_summary')):
            emb = resume['embedding_summary']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            sum_results = jobs_table.search(
                emb,
                vector_column_name="embedding_description"
            ).limit(limit).to_list()
            results.extend(sum_results)
        
        # 3. Search by resume skills against job skills
        if is_valid_embedding(resume.get('embedding_skills')):
            emb = resume['embedding_skills']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            skill_results = jobs_table.search(
                emb,
                vector_column_name="embedding_skills"
            ).limit(limit).to_list()
            results.extend(skill_results)
        
        # 4. Combined embedding search
        if is_valid_embedding(resume.get('embedding_combined')):
            emb = resume['embedding_combined']
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            combined_results = jobs_table.search(
                emb,
                vector_column_name="embedding_full"
            ).limit(limit).to_list()
            results.extend(combined_results)
        
        # Deduplicate by job_id and return top matches
        seen = set()
        unique_results = []
        for r in results:
            if r['job_id'] not in seen:
                seen.add(r['job_id'])
                unique_results.append(r)
                if len(unique_results) >= limit:
                    break
        
        return unique_results
    
    def get_all_resumes(self) -> List[Dict]:
        """Get all resumes from database"""
        try:
            table = self.db.open_table("resumes")
            return table.to_pandas().to_dict('records')
        except:
            logger.warning("Resumes table not found")
            return []
    
    def get_all_jobs(self) -> List[Dict]:
        """Get all jobs from database"""
        try:
            table = self.db.open_table("jobs")
            return table.to_pandas().to_dict('records')
        except:
            logger.warning("Jobs table not found")
            return []
    
    def get_resume_by_id(self, resume_id: str) -> Optional[Dict]:
        """Get a specific resume by ID"""
        try:
            table = self.db.open_table("resumes")
            df = table.to_pandas()
            resume = df[df['resume_id'] == resume_id]
            if not resume.empty:
                return resume.iloc[0].to_dict()
        except:
            logger.warning(f"Could not retrieve resume {resume_id}")
        return None
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Get a specific job by ID"""
        try:
            table = self.db.open_table("jobs")
            df = table.to_pandas()
            job = df[df['job_id'] == job_id]
            if not job.empty:
                return job.iloc[0].to_dict()
        except:
            logger.warning(f"Could not retrieve job {job_id}")
        return None
    
    def delete_job(self, job_id: str) -> bool:
        """Delete a specific job by ID and cascade delete associated matches"""
        try:
            # First, delete all matches associated with this job
            deleted_matches = self.delete_matches_for_job(job_id)
            if deleted_matches > 0:
                logger.info(f"Deleted {deleted_matches} matches associated with job {job_id}")
            
            # Now delete the job itself
            table = self.db.open_table("jobs")
            df = table.to_pandas()
            
            # Filter out the job to delete
            df_filtered = df[df['job_id'] != job_id]
            
            if len(df_filtered) < len(df):
                # Job was found and filtered out
                # Recreate the table with filtered data
                self.drop_table("jobs")
                if len(df_filtered) > 0:
                    # Re-add remaining jobs
                    self.db.create_table("jobs", data=df_filtered.to_dict('records'))
                else:
                    # Table is now empty, just recreate schema
                    self.db.create_table("jobs", schema=JobPosting.to_arrow_schema())
                
                logger.info(f"Deleted job {job_id}")
                return True
            else:
                logger.warning(f"Job {job_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Could not delete job {job_id}: {e}")
            return False
    
    def delete_matches_for_job(self, job_id: str) -> int:
        """Delete all matches associated with a specific job"""
        try:
            # Check if matches table exists
            if "matches" not in self.db.table_names():
                return 0
                
            table = self.db.open_table("matches")
            df = table.to_pandas()
            
            # If table is empty, return 0
            if len(df) == 0:
                return 0
            
            # Count matches to delete
            matches_to_delete = len(df[df['job_id'] == job_id])
            
            if matches_to_delete > 0:
                # Filter out matches for this job
                df_filtered = df[df['job_id'] != job_id]
                
                # Recreate the table with filtered data
                self.drop_table("matches")
                if len(df_filtered) > 0:
                    # Re-add remaining matches
                    self.db.create_table("matches", data=df_filtered.to_dict('records'))
                else:
                    # Table is now empty, just recreate schema
                    self.db.create_table("matches", schema=JobMatch.to_arrow_schema())
                
                logger.info(f"Deleted {matches_to_delete} matches for job {job_id}")
            
            return matches_to_delete
            
        except Exception as e:
            logger.error(f"Could not delete matches for job {job_id}: {e}")
            return 0
    
    def delete_matches_for_resume(self, resume_id: str) -> int:
        """Delete all matches associated with a specific resume"""
        try:
            if not self.table_exists("matches"):
                return 0
                
            table = self.db.open_table("matches")
            df = table.to_pandas()
            
            # Count matches to delete
            matches_to_delete = len(df[df['resume_id'] == resume_id])
            
            if matches_to_delete > 0:
                # Filter out matches for this resume
                df_filtered = df[df['resume_id'] != resume_id]
                
                # Recreate the table with filtered data
                self.drop_table("matches")
                if len(df_filtered) > 0:
                    # Re-add remaining matches
                    self.db.create_table("matches", data=df_filtered.to_dict('records'))
                else:
                    # Table is now empty, just recreate schema
                    self.db.create_table("matches", schema=JobMatch.to_arrow_schema())
                
                logger.info(f"Deleted {matches_to_delete} matches for resume {resume_id}")
            
            return matches_to_delete
            
        except Exception as e:
            logger.error(f"Could not delete matches for resume {resume_id}: {e}")
            return 0
    
    def delete_resume(self, resume_id: str) -> bool:
        """Delete a specific resume by ID and cascade delete associated matches"""
        try:
            # First, delete all matches associated with this resume
            deleted_matches = self.delete_matches_for_resume(resume_id)
            if deleted_matches > 0:
                logger.info(f"Deleted {deleted_matches} matches associated with resume {resume_id}")
            
            # Now delete the resume itself
            table = self.db.open_table("resumes")
            df = table.to_pandas()
            
            # Filter out the resume to delete
            df_filtered = df[df['resume_id'] != resume_id]
            
            if len(df_filtered) < len(df):
                # Resume was found and filtered out
                # Recreate the table with filtered data
                self.drop_table("resumes")
                if len(df_filtered) > 0:
                    # Re-add remaining resumes
                    self.db.create_table("resumes", data=df_filtered.to_dict('records'))
                else:
                    # Table is now empty, just recreate schema
                    self.db.create_table("resumes", schema=Resume.to_arrow_schema())
                
                logger.info(f"Deleted resume {resume_id}")
                return True
            else:
                logger.warning(f"Resume {resume_id} not found for deletion")
                return False
                
        except Exception as e:
            logger.error(f"Could not delete resume {resume_id}: {e}")
            return False
    
    # -------------------------
    # Match Operations
    # -------------------------
    
    def clear_all_matches(self):
        """Clear all existing matches from the database"""
        try:
            # Drop and recreate the matches table
            if "matches" in self.db.table_names():
                self.recreate_table("matches")
                logger.info("Cleared all existing matches")
        except Exception as e:
            logger.error(f"Error clearing matches: {e}")
    
    # -------------------------
    # Analytics Report Operations
    # -------------------------
    
    def save_analytics_report(self, report_data: Dict) -> bool:
        """Save or update an analytics report for a resume
        
        Args:
            report_data: Dictionary containing report data with resume_id as key
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create table if it doesn't exist
            if "analytics_reports" not in self.db.table_names():
                # Create with dummy data to establish schema
                dummy_report = AnalyticsReport(
                    resume_id="dummy",
                    report_timestamp=datetime.now().isoformat(),
                    generation_date=datetime.now().strftime("%Y-%m-%d"),
                    total_jobs_analyzed=0,
                    total_matches_analyzed=0,
                    avg_match_score=0.0,
                    match_distribution="{}",
                    top_demanded_skills="[]",
                    skill_gaps="[]",
                    skill_strengths="[]",
                    top_job_titles="[]",
                    market_insights="",
                    recommendations="",
                    user_preferences="{}",
                    full_report_json="{}"
                )
                self.db.create_table("analytics_reports", data=[dummy_report])
                # Delete the dummy record
                table = self.db.open_table("analytics_reports")
                table.delete("resume_id = 'dummy'")
            
            table = self.db.open_table("analytics_reports")
            
            # Check if report already exists for this resume
            existing_df = table.to_pandas()
            resume_id = report_data.get('resume_id')
            
            if resume_id in existing_df['resume_id'].values:
                # Delete existing report (will be replaced)
                table.delete(f"resume_id = '{resume_id}'")
                logger.info(f"Replacing existing analytics report for resume {resume_id}")
            
            # Create new report record
            report = AnalyticsReport(**report_data)
            table.add([report])
            
            logger.info(f"Saved analytics report for resume {resume_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving analytics report: {e}")
            return False
    
    def get_analytics_report(self, resume_id: str) -> Optional[Dict]:
        """Get analytics report for a specific resume
        
        Args:
            resume_id: ID of the resume
            
        Returns:
            Report data as dictionary or None if not found
        """
        try:
            if "analytics_reports" not in self.db.table_names():
                return None
                
            table = self.db.open_table("analytics_reports")
            df = table.to_pandas()
            
            report = df[df['resume_id'] == resume_id]
            if not report.empty:
                return report.iloc[0].to_dict()
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving analytics report: {e}")
            return None
    
    def get_all_analytics_reports(self) -> List[Dict]:
        """Get all analytics reports
        
        Returns:
            List of report dictionaries
        """
        try:
            if "analytics_reports" not in self.db.table_names():
                return []
                
            table = self.db.open_table("analytics_reports")
            return table.to_pandas().to_dict('records')
            
        except Exception as e:
            logger.error(f"Error retrieving analytics reports: {e}")
            return []
    
    def delete_analytics_report(self, resume_id: str) -> bool:
        """Delete analytics report for a specific resume
        
        Args:
            resume_id: ID of the resume
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            if "analytics_reports" not in self.db.table_names():
                return False
                
            table = self.db.open_table("analytics_reports")
            table.delete(f"resume_id = '{resume_id}'")
            logger.info(f"Deleted analytics report for resume {resume_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting analytics report: {e}")
            return False
    
    
    def _format_llm_assessment(self, reasoning: str, recommendations: List[str]) -> str:
        """Format LLM reasoning and recommendations into a single assessment string"""
        if not reasoning and not recommendations:
            return ""
        
        assessment = reasoning or ""
        if recommendations:
            if assessment:
                assessment += "\n\nRecommendations:\n"
            assessment += "\n".join(f"• {rec}" for rec in recommendations)
        
        return assessment
    
    def add_match(self, match_data: Dict[str, Any]) -> str:
        """Add a job-resume match with detailed scoring"""
        
        timestamp = datetime.now().isoformat()
        match_id = f"match_{match_data['job_id']}_{match_data['resume_id']}_{timestamp}"
        
        record = {
            "match_id": match_id,
            "job_id": match_data['job_id'],
            "resume_id": match_data['resume_id'],
            "overall_score": match_data['overall_score'],
            "semantic_score": match_data.get('semantic_score', 0.0),
            "title_match_score": match_data.get('title_match_score', 0.0),
            "skills_score": match_data.get('skills_score', 0.0),
            "experience_score": match_data.get('experience_score', 0.0),
            "requirements_score": match_data.get('requirements_score', 0.0),
            "education_score": match_data.get('education_score'),  # Added
            "location_score": match_data.get('location_score'),  # Added
            "salary_score": match_data.get('salary_score'),  # Added
            "summary_to_description_score": match_data.get('summary_to_description_score'),
            "experience_to_requirements_score": match_data.get('experience_to_requirements_score'),
            "skills_to_skills_score": match_data.get('skills_to_skills_score'),
            "location_preference_met": match_data.get('location_preference_met'),  # Added
            "remote_preference_met": match_data.get('remote_preference_met'),  # Added
            "salary_match": json.dumps(match_data.get('salary_match', {})) if match_data.get('salary_match') else None,  # Added
            "llm_score": match_data.get('llm_score'),
            # Combine llm_reasoning and recommendations into llm_assessment
            "llm_assessment": self._format_llm_assessment(
                match_data.get('llm_reasoning') or match_data.get('llm_assessment'),
                match_data.get('llm_recommendations', [])
            ),
            "match_reasons": json.dumps(match_data.get('match_reasons', [])),
            "skills_matched": json.dumps(match_data.get('skills_matched', [])) if match_data.get('skills_matched') else None,
            "skills_gap": json.dumps(match_data.get('skills_gap', [])) if match_data.get('skills_gap') else None,
            "requirements_matched": json.dumps(match_data.get('requirements_matched', [])) if match_data.get('requirements_matched') else None,
            "requirements_gap": json.dumps(match_data.get('requirements_gap', [])) if match_data.get('requirements_gap') else None,
            "match_timestamp": timestamp,
            "pipeline_run_id": match_data.get('pipeline_run_id')
        }
        
        table = self.db.open_table("matches")
        table.add([record])
        logger.info(f"Added match {match_id}")
        
        return match_id
    
    def get_matches_for_job(self, job_id: str) -> List[Dict]:
        """Get all matches for a specific job"""
        try:
            table = self.db.open_table("matches")
            df = table.to_pandas()
            matches = df[df['job_id'] == job_id]
            return matches.to_dict('records')
        except:
            logger.warning(f"Could not retrieve matches for job {job_id}")
            return []
    
    def get_matches_for_resume(self, resume_id: str) -> List[Dict]:
        """Get all matches for a specific resume"""
        try:
            table = self.db.open_table("matches")
            df = table.to_pandas()
            matches = df[df['resume_id'] == resume_id]
            return matches.to_dict('records')
        except:
            logger.warning(f"Could not retrieve matches for resume {resume_id}")
            return []
    
    def get_top_matches(self, limit: int = 1000) -> List[Dict]:
        """Get top matches by overall score, removing duplicates"""
        try:
            table = self.db.open_table("matches")
            df = table.to_pandas()
            
            # Remove duplicates based on job_id and resume_id, keeping the highest score
            df = df.sort_values('overall_score', ascending=False)
            df = df.drop_duplicates(subset=['job_id', 'resume_id'], keep='first')
            
            top_matches = df.nlargest(limit, 'overall_score')
            return top_matches.to_dict('records')
        except:
            logger.warning("Could not retrieve top matches")
            return []
    
    def save_matches(self, matches: List[Dict], pipeline_run_id: str = None) -> int:
        """Save multiple matches from matching pipeline, avoiding duplicates"""
        saved_count = 0
        
        try:
            table = self.db.open_table("matches")
            existing_df = table.to_pandas()
            
            for match in matches:
                job_id = match.get('job_id')
                resume_id = match.get('resume_id')
                
                # Check if this match already exists
                existing = existing_df[
                    (existing_df['job_id'] == job_id) & 
                    (existing_df['resume_id'] == resume_id)
                ]
                
                # If exists and new score is higher, delete old one
                if not existing.empty:
                    old_score = existing.iloc[0]['overall_score']
                    new_score = match.get('overall_score', 0)
                    
                    if new_score > old_score:
                        # Delete old match (we'll add the new one below)
                        logger.info(f"Updating match {job_id} x {resume_id}: {old_score:.3f} -> {new_score:.3f}")
                    else:
                        # Skip this match as existing one is better
                        logger.debug(f"Skipping match {job_id} x {resume_id}: existing score {old_score:.3f} >= new {new_score:.3f}")
                        continue
                
                # Add the match
                try:
                    self.add_match(match)
                    saved_count += 1
                except Exception as e:
                    logger.warning(f"Failed to save match: {e}")
        
        except Exception as e:
            logger.error(f"Error saving matches: {e}")
        
        return saved_count


# -------------------------
# Utility Functions
# -------------------------

def test_database():
    """Test database operations"""
    db = graphDB(reset_if_exists=True)
    
    # Test job addition
    test_job = {
        'text': 'Software Engineer at Tech Corp. We are looking for a Python developer with 5 years experience.',
        'title': 'Software Engineer',
        'company': 'Tech Corp',
        'description': 'We are looking for a talented engineer to join our team.',
        'requirements': ['5 years Python experience', 'Bachelor degree in CS'],
        'skills': ['Python', 'Django', 'PostgreSQL']
    }
    
    job_id = db.add_job(test_job, 'python developer', 'remote')
    print(f"Added job: {job_id}")
    
    # Test resume addition
    test_resume = {
        'contact_info': {
            'name': 'John Doe',
            'email': 'john@example.com',
            'location': 'San Francisco, CA'
        },
        'full_text': 'John Doe - Software Engineer with 6 years of Python development experience...',
        'professional_summary': 'Experienced software engineer specializing in Python and web development.',
        'work_experience': [
            {
                'title': 'Senior Python Developer',
                'company': 'Previous Corp',
                'responsibilities': ['Developed Django applications', 'Managed PostgreSQL databases']
            }
        ],
        'technical_skills': ['Python', 'Django', 'Flask', 'PostgreSQL', 'Docker'],
        'frameworks_tools': ['Git', 'AWS', 'Jenkins'],
        'years_of_experience': 6,
        'filename': 'john_doe_resume.pdf'
    }
    
    resume_id = db.add_resume(test_resume)
    print(f"Added resume: {resume_id}")
    
    # Test search
    matches = db.search_jobs_by_resume(resume_id, limit=5)
    print(f"Found {len(matches)} matching jobs for resume")
    
    # Test match addition
    match_data = {
        'job_id': job_id,
        'resume_id': resume_id,
        'overall_score': 0.85,
        'semantic_score': 0.9,
        'skills_score': 0.8,
        'experience_score': 0.85,
        'match_reasons': ['Strong Python experience', 'Meets experience requirements'],
        'skills_matched': ['Python', 'Django', 'PostgreSQL']
    }
    
    match_id = db.add_match(match_data)
    print(f"Added match: {match_id}")
    
    print("\nDatabase test completed successfully!")


if __name__ == "__main__":
    test_database()

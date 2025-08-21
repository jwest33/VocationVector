"""
Simplified consolidated pipeline for job matching system
Removes redundant "enhanced" naming and consolidates functionality
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

# LangGraph for workflow orchestration
from langgraph.graph import StateGraph, END

# Import our components
from graph.crawler import BulkJobsCrawler
from graph.database import graphDB
from graph.llm_server import LLMServerManager
from graph.settings import get_settings
from graph.cache_manager import CacheManager
from graph.embeddings import JobEmbeddings

# Import processing nodes
from graph.nodes.resume_processing import resume_processing_node
from graph.nodes.job_processing import job_processing_node
from graph.nodes.job_matching import job_matching_node
from graph.nodes.enhanced_matching import enhanced_matching_node

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------
# Pipeline Configuration
# -------------------------

class PipelineMode(Enum):
    """Pipeline execution modes"""
    PROCESS_JOBS = "process_jobs"  # Crawl and process jobs
    PROCESS_RESUMES = "process_resumes"  # Process resumes only
    MATCH_ONLY = "match_only"  # Match existing data
    FULL_PIPELINE = "full_pipeline"  # Complete pipeline


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    
    # Mode
    mode: PipelineMode = PipelineMode.FULL_PIPELINE
    
    # Job crawling settings
    job_query: str = "python developer"
    job_location: str = "remote"
    max_jobs: int = 20
    crawl_headless: bool = True
    
    # Resume processing settings
    resume_directory: str = "data/resumes"
    resume_files: Optional[List[str]] = None
    resume_path: Optional[str] = None  # Single resume file path
    
    # Matching settings
    use_llm_matching: bool = True
    top_k_matches: int = 1000  # Essentially unlimited - filter by score instead
    min_match_score: float = 0.3
    clear_all_matches: bool = False  # Set to True for manual matching
    
    # Output settings
    output_directory: str = "data/pipeline_output"
    save_intermediate: bool = True
    
    # Database settings
    use_database: bool = True
    db_path: Optional[str] = None
    
    # LLM settings
    llm_model: str = os.getenv("LLM_MODEL", "qwen3-4b-instruct-2507-f16")
    llm_base_url: str = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    llm_timeout: int = 240
    
    # Server management
    auto_start_server: bool = True
    stop_server_on_complete: bool = False
    
    # Caching settings
    use_cache: bool = True
    cache_directory: str = "data/.cache"
    force_refresh: bool = False
    
    # Processing options
    verbose: bool = False
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PipelineConfig':
        """Create config from dictionary"""
        if "mode" in config_dict and isinstance(config_dict["mode"], str):
            config_dict["mode"] = PipelineMode(config_dict["mode"])
        
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__annotations__})
    
    @classmethod
    def from_settings(cls, **overrides) -> 'PipelineConfig':
        """Create config from central settings with optional overrides"""
        settings = get_settings()
        
        config = cls(
            max_jobs=settings.crawler.default_max_jobs,
            crawl_headless=settings.crawler.headless,
            resume_directory=settings.pipeline.resume_directory,
            use_llm_matching=settings.pipeline.use_llm_matching,
            top_k_matches=settings.pipeline.top_k_matches,
            min_match_score=settings.pipeline.min_match_score,
            output_directory=settings.pipeline.output_directory,
            save_intermediate=settings.pipeline.save_intermediate_results,
            llm_model=settings.llm.llm_name,
            llm_base_url=settings.llm.base_url,
            llm_timeout=settings.llm.timeout,
            verbose=settings.pipeline.verbose,
            auto_start_server=settings.llm.auto_start_server,
            stop_server_on_complete=settings.llm.stop_server_on_complete,
        )
        
        # Apply overrides
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result["mode"] = self.mode.value
        return result


# -------------------------
# Pipeline Nodes
# -------------------------

async def crawl_and_process_jobs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Crawl and process jobs - always do both together"""
    
    config = state["config"]
    logger.info(f"Crawling jobs: query='{config.job_query}', location='{config.job_location}'")
    
    try:
        # Crawl jobs
        crawler = BulkJobsCrawler(
            headless=config.crawl_headless,
            data_dir=f"{config.output_directory}/jobs"
        )
        
        result = await crawler.get_all_jobs_expanded(
            query=config.job_query,
            location=config.job_location,
            max_jobs=config.max_jobs
        )
        
        if "error" in result:
            state["errors"].append(f"Crawling error: {result['error']}")
            state["raw_jobs"] = []
            state["processed_jobs"] = []
            return state
        
        jobs = result.get("jobs", [])
        
        # Store raw jobs in state for LLM processing
        state["raw_jobs"] = jobs
        
        # Call the job processing node to use LLM
        logger.info(f"Processing {len(jobs)} crawled jobs with LLM")
        state = job_processing_node(state)
        
        # Get the processed jobs with LLM-extracted data 
        processed_templates = state.get("processed_jobs", [])
        
        # Merge the processed data with the original job data
        processed_jobs = []
        for i, (orig_job, template) in enumerate(zip(jobs, processed_templates)):
            # Extract and format skills properly
            skills_list = []
            for skill in template.get('technical_skills', []):
                if isinstance(skill, dict):
                    skill_name = skill.get('skill_name', skill.get('skill', ''))
                    if skill_name:
                        skills_list.append(skill_name)
                elif isinstance(skill, str):
                    skills_list.append(skill)
            
            # Extract requirements from experience requirements
            requirements_list = []
            for exp_req in template.get('experience_requirements', []):
                if isinstance(exp_req, dict):
                    desc = exp_req.get('description', '')
                    if desc:
                        requirements_list.append(desc)
            
            # Add education requirements to requirements
            for edu_req in template.get('education_requirements', []):
                if isinstance(edu_req, dict):
                    degree = edu_req.get('degree_level', '')
                    field = edu_req.get('field_of_study', '')
                    if degree:
                        req_text = f"{degree}"
                        if field:
                            req_text += f" in {field}"
                        requirements_list.append(req_text)
            
            # Extract benefits properly
            compensation = template.get('compensation', {})
            benefits_list = compensation.get('benefits_required', []) if isinstance(compensation, dict) else []
            
            # Build a proper description from summary and key responsibilities
            description_parts = []
            if template.get('summary'):
                description_parts.append(template.get('summary'))
            
            # Add key responsibilities to description if no separate responsibilities field
            key_responsibilities = template.get('key_responsibilities', [])
            if key_responsibilities and isinstance(key_responsibilities, list):
                description_parts.append("\n\nKey Responsibilities:")
                for resp in key_responsibilities[:5]:  # Limit to first 5
                    if resp:
                        description_parts.append(f"• {resp}")
            
            description = '\n'.join(description_parts) if description_parts else orig_job.get('text', '')[:500]
            
            # Combine original job data with processed template
            merged_job = {
                **orig_job,  # Contains text, job_index, posted_date, etc.
                'title': template.get('title', 'Unknown Position'),
                'company': template.get('metadata', {}).get('company', 'Unknown Company') if isinstance(template.get('metadata'), dict) else 'Unknown Company',
                'description': description,
                'skills': skills_list,
                'requirements': requirements_list if requirements_list else ['No specific requirements listed'],
                'responsibilities': key_responsibilities if key_responsibilities else ['See job description'],
                'benefits': benefits_list if benefits_list else ['Benefits package available'],
                'salary_min': compensation.get('minimum_salary') if isinstance(compensation, dict) else None,
                'salary_max': compensation.get('maximum_salary') if isinstance(compensation, dict) else None,
                'employment_type': template.get('employment_type'),
                'remote_policy': template.get('location', {}).get('work_arrangement') if isinstance(template.get('location'), dict) else None,
                'vocation_template': template,  # Store the full template for advanced features
                'years_experience_required': template.get('total_years_experience'),
                'education_requirements': template.get('education_requirements', []),
                'equity': compensation.get('equity_expectation') if isinstance(compensation, dict) else None,
                'bonus': compensation.get('bonus_structure') if isinstance(compensation, dict) else None,
                'team_size': template.get('culture_fit', {}).get('team_size_preference') if isinstance(template.get('culture_fit'), dict) else None,
                'growth_opportunities': template.get('culture_fit', {}).get('career_growth_importance') if isinstance(template.get('culture_fit'), dict) else None,
                'start_date': template.get('start_date')
            }
            processed_jobs.append(merged_job)
        
        logger.info(f"Crawled and processed {len(processed_jobs)} jobs")
        
        # Save to database
        if config.use_database and processed_jobs:
            try:
                db = state.get("database")
                if not db:
                    db = graphDB(db_path=config.db_path)
                    state["database"] = db
                
                count, updated_job_ids = db.add_jobs_batch(
                    processed_jobs,
                    search_query=config.job_query,
                    search_location=config.job_location
                )
                logger.info(f"Added {count} jobs to database")
                # Track updated job IDs for incremental matching
                state["updated_job_ids"] = updated_job_ids
            except Exception as e:
                logger.warning(f"Failed to save jobs to database: {e}")
        
        # Save intermediate results
        if config.save_intermediate:
            output_file = Path(config.output_directory) / "jobs" / "processed_jobs.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    "jobs": processed_jobs,
                    "query": config.job_query,
                    "location": config.job_location,
                    "timestamp": datetime.now().isoformat()
                }, f, indent=2)
        
    except Exception as e:
        logger.error(f"Job crawling/processing failed: {e}")
        state["errors"].append(f"Job processing exception: {str(e)}")
        state["raw_jobs"] = []
        state["processed_jobs"] = []
    
    return state


def process_resumes_node_simple(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process resumes from files or database"""
    
    config = state["config"]
    logger.info(f"Processing resumes from: {config.resume_directory}")
    
    # Initialize database if needed
    db = state.get("database")
    if not db and config.use_database:
        db = graphDB(db_path=config.db_path)
        state["database"] = db
    
    # Try to load from database first
    if db:
        try:
            resumes = db.get_all_resumes()
            if resumes:
                logger.info(f"Loaded {len(resumes)} resumes from database")
                state["processed_resumes"] = resumes
                
                # Create templates for matching
                resume_templates = []
                for resume in resumes:
                    template = _create_resume_template(resume)
                    resume_templates.append(template)
                
                state["resume_templates"] = resume_templates
                return state
        except Exception as e:
            logger.warning(f"Could not load resumes from database: {e}")
    
    # Process from files
    try:
        # Get resume files
        if hasattr(config, 'resume_path') and config.resume_path:
            # Process single resume file
            resume_files = [config.resume_path]
            logger.info(f"Processing single resume: {config.resume_path}")
        elif config.resume_files:
            resume_files = config.resume_files
        else:
            resume_dir = Path(config.resume_directory)
            resume_files = list(resume_dir.glob("*.txt")) + \
                          list(resume_dir.glob("*.pdf")) + \
                          list(resume_dir.glob("*.docx"))
            resume_files = [str(f) for f in resume_files]
        
        if not resume_files:
            logger.warning("No resume files found")
            state["processed_resumes"] = []
            state["resume_templates"] = []
            return state
        
        logger.info(f"Found {len(resume_files)} resume files")
        
        # Process resumes using simple extraction
        from graph.skill_extractor import extract_skills_from_text
        
        processed_resumes = []
        resume_templates = []
        
        for file_path in resume_files:
            try:
                # Read file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Extract skills
                skills = extract_skills_from_text(text)
                
                # Create resume record
                resume_data = {
                    "resume_id": f"resume_{datetime.now().isoformat()}_{Path(file_path).stem}",
                    "filename": Path(file_path).name,
                    "full_text": text,
                    "skills": skills,
                    "name": _extract_name_from_text(text),
                    "email": _extract_email_from_text(text),
                    "phone": _extract_phone_from_text(text),
                    "summary": text[:500]
                }
                
                processed_resumes.append(resume_data)
                
                # Create template
                template = _create_resume_template(resume_data)
                resume_templates.append(template)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        state["processed_resumes"] = processed_resumes
        state["resume_templates"] = resume_templates
        
        # Save to database
        if config.use_database and processed_resumes:
            try:
                count = 0
                for resume in processed_resumes:
                    try:
                        db.add_resume(resume)
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to save resume to database: {e}")
                logger.info(f"Added {count} resumes to database")
            except Exception as e:
                logger.warning(f"Failed to save resumes to database: {e}")
        
        # Save intermediate results
        if config.save_intermediate:
            output_file = Path(config.output_directory) / "resumes" / "processed_resumes.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    "resumes": processed_resumes,
                    "templates": resume_templates
                }, f, indent=2)
        
        logger.info(f"Processed {len(processed_resumes)} resumes")
        
    except Exception as e:
        logger.error(f"Resume processing failed: {e}")
        state["errors"].append(f"Resume processing exception: {str(e)}")
        state["processed_resumes"] = []
        state["resume_templates"] = []
    
    return state


def match_jobs_node_simple(state: Dict[str, Any]) -> Dict[str, Any]:
    """Match resumes to jobs"""
    
    config = state["config"]
    
    # Get jobs and resumes
    jobs = state.get("processed_jobs", [])
    resumes = state.get("processed_resumes", [])
    
    # For match_only mode, load from database
    if config.mode == PipelineMode.MATCH_ONLY:
        db = state.get("database")
        if not db:
            db = graphDB(db_path=config.db_path)
            state["database"] = db
        
        if not jobs:
            jobs = db.get_all_jobs()
            logger.info(f"Loaded {len(jobs)} jobs from database")
        
        if not resumes:
            resumes = db.get_all_resumes()
            logger.info(f"Loaded {len(resumes)} resumes from database")
    
    if not jobs or not resumes:
        logger.warning(f"Cannot match: {len(jobs)} jobs, {len(resumes)} resumes")
        state["matching_results"] = []
        state["best_matches"] = []
        return state
    
    logger.info(f"Matching {len(resumes)} resumes to {len(jobs)} jobs")
    
    try:
        all_matches = []
        
        for resume in resumes:
            # Get resume skills
            resume_skills = resume.get('skills', [])
            if isinstance(resume_skills, str):
                try:
                    resume_skills = json.loads(resume_skills)
                except:
                    resume_skills = []
            
            # Ensure skills are strings, not dicts
            if resume_skills and isinstance(resume_skills[0], dict):
                resume_skills = [s.get('name', s.get('skill', str(s))) for s in resume_skills if isinstance(s, dict)]
            
            for job in jobs:
                try:
                    # Get job skills
                    job_skills = []
                    if 'vocation_template' in job:
                        template = job['vocation_template']
                        if isinstance(template, str):
                            template = json.loads(template)
                        job_skills = template.get('technical_skills', [])
                    
                    # Ensure job skills are strings, not dicts
                    if job_skills and isinstance(job_skills[0], dict):
                        job_skills = [s.get('name', s.get('skill', str(s))) for s in job_skills if isinstance(s, dict)]
                    
                    # Calculate match score
                    if resume_skills and job_skills:
                        common_skills = set(resume_skills) & set(job_skills)
                        skill_score = len(common_skills) / max(len(resume_skills), len(job_skills))
                    else:
                        skill_score = 0
                    
                    if skill_score >= config.min_match_score:
                        match = {
                            'resume_id': resume.get('resume_id', ''),
                            'resume_name': resume.get('name', 'Unknown'),
                            'job_id': job.get('job_id', f"job_{job.get('job_index', 0)}"),
                            'job_title': job.get('title', 'Unknown Position'),
                            'company': job.get('company', 'Unknown'),
                            'overall_score': skill_score,
                            'skills_score': skill_score,
                            'matched_skills': list(common_skills) if resume_skills and job_skills else [],
                            'match_reasons': []
                        }
                        
                        # Add match reasons
                        if skill_score > 0.5:
                            match['match_reasons'].append(f"Strong skill match ({len(common_skills)} skills)")
                        elif skill_score > 0.3:
                            match['match_reasons'].append(f"Good skill match ({len(common_skills)} skills)")
                        
                        if 'remote' in job.get('location', '').lower():
                            match['match_reasons'].append("Remote position")
                        
                        all_matches.append(match)
                    
                except Exception as e:
                    logger.error(f"Error matching resume to job: {e}")
        
        # Sort matches by score
        all_matches.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Keep top matches
        best_matches = all_matches[:config.top_k_matches]
        
        state["matching_results"] = all_matches
        state["best_matches"] = best_matches
        
        logger.info(f"Found {len(all_matches)} total matches, kept {len(best_matches)} best")
        
        # Save to database
        if config.use_database and best_matches:
            try:
                db = state.get("database")
                if not db:
                    db = graphDB(db_path=config.db_path)
                    state["database"] = db
                
                pipeline_run_id = f"pipeline_{datetime.now().isoformat()}"
                count = db.save_matches(best_matches, pipeline_run_id=pipeline_run_id)
                logger.info(f"Saved {count} matches to database")
            except Exception as e:
                logger.warning(f"Failed to save matches to database: {e}")
        
        # Save intermediate results
        if config.save_intermediate:
            output_file = Path(config.output_directory) / "matches" / "matching_results.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    "matching_results": all_matches[:50],  # Top 50
                    "best_matches": best_matches
                }, f, indent=2)
        
    except Exception as e:
        logger.error(f"Job matching failed: {e}")
        state["errors"].append(f"Matching exception: {str(e)}")
        state["matching_results"] = []
        state["best_matches"] = []
    
    return state


def finalize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Finalize pipeline and prepare output"""
    
    config = state["config"]
    state["end_time"] = datetime.now()
    
    if state.get("start_time"):
        duration = (state["end_time"] - state["start_time"]).total_seconds()
        state["diagnostics"]["total_duration"] = duration
        logger.info(f"Pipeline completed in {duration:.2f} seconds")
    
    # Create final output
    output = {
        "timestamp": state["end_time"].isoformat(),
        "config": config.to_dict(),
        "statistics": {
            "jobs_processed": len(state.get("processed_jobs", [])),
            "resumes_processed": len(state.get("processed_resumes", [])),
            "matches_found": len(state.get("best_matches", [])),
            "errors": len(state.get("errors", []))
        },
        "best_matches": state.get("best_matches", [])[:20],
        "diagnostics": state.get("diagnostics", {}),
        "errors": state.get("errors", [])
    }
    
    # Save final output
    output_file = Path(config.output_directory) / "pipeline_output.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Pipeline output saved to: {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Jobs processed: {output['statistics']['jobs_processed']}")
    print(f"Resumes processed: {output['statistics']['resumes_processed']}")
    print(f"Matches found: {output['statistics']['matches_found']}")
    
    if output["best_matches"]:
        print(f"\nTop 3 Matches:")
        for i, match in enumerate(output["best_matches"][:3], 1):
            print(f"{i}. {match.get('resume_name', 'Unknown')} -> {match.get('job_title', 'Unknown')} at {match.get('company', '')}")
            print(f"   Score: {match.get('overall_score', 0):.1%}")
    
    if output["errors"]:
        print(f"\n⚠️ Errors encountered: {len(output['errors'])}")
        for error in output["errors"][:3]:
            print(f"  - {error}")
    
    return state


# -------------------------
# Helper Functions
# -------------------------

def _create_resume_template(resume: Dict) -> Dict:
    """Create a template for matching from resume data"""
    skills = resume.get('skills', [])
    if isinstance(skills, str):
        try:
            skills = json.loads(skills)
        except:
            skills = []
    
    return {
        'candidate_profile': {
            'name': resume.get('name', 'Unknown'),
            'email': resume.get('email', ''),
            'phone': resume.get('phone', ''),
            'summary': resume.get('summary', '')
        },
        'skills': skills,
        'resume_id': resume.get('resume_id'),
        'full_text': resume.get('full_text', '')
    }


def _extract_name_from_text(text: str) -> str:
    """Extract name from resume text"""
    lines = text.split('\n')
    for line in lines[:5]:
        line = line.strip()
        if line and len(line.split()) <= 4 and not any(char.isdigit() for char in line):
            return line
    return "Unknown"


def _extract_email_from_text(text: str) -> str:
    """Extract email from text"""
    import re
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    match = re.search(email_pattern, text)
    return match.group(0) if match else ""


def _extract_phone_from_text(text: str) -> str:
    """Extract phone number from text"""
    import re
    phone_pattern = r'[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else ""


# -------------------------
# Pipeline Builder
# -------------------------

class JobMatchingPipeline:
    """Main pipeline class for job matching system"""
    
    def __init__(self, config: Optional[Union[Dict, PipelineConfig]] = None):
        """Initialize pipeline with configuration"""
        
        if config is None:
            self.config = PipelineConfig()
        elif isinstance(config, dict):
            self.config = PipelineConfig.from_dict(config)
        else:
            self.config = config
        
        self.graph = self._build_graph()
        self.server_manager = None
        
        # Initialize server manager if auto-start is enabled
        if self.config.auto_start_server:
            self.server_manager = LLMServerManager()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create graph
        workflow = StateGraph(dict)
        
        # Add nodes based on mode
        mode = self.config.mode
        
        if mode in [PipelineMode.PROCESS_JOBS, PipelineMode.FULL_PIPELINE]:
            workflow.add_node("process_jobs", self._wrap_async_node(crawl_and_process_jobs_node))
        
        if mode in [PipelineMode.PROCESS_RESUMES, PipelineMode.FULL_PIPELINE]:
            workflow.add_node("process_resumes", resume_processing_node)
        
        if mode in [PipelineMode.MATCH_ONLY, PipelineMode.FULL_PIPELINE]:
            # Use enhanced matching for comprehensive feature comparison
            workflow.add_node("match_jobs", enhanced_matching_node)
        
        workflow.add_node("finalize", finalize_node)
        
        # Set entry point and edges based on mode
        if mode == PipelineMode.PROCESS_JOBS:
            workflow.set_entry_point("process_jobs")
            workflow.add_edge("process_jobs", "finalize")
        
        elif mode == PipelineMode.PROCESS_RESUMES:
            workflow.set_entry_point("process_resumes")
            workflow.add_edge("process_resumes", "finalize")
        
        elif mode == PipelineMode.MATCH_ONLY:
            workflow.set_entry_point("match_jobs")
            workflow.add_edge("match_jobs", "finalize")
        
        elif mode == PipelineMode.FULL_PIPELINE:
            workflow.set_entry_point("process_jobs")
            workflow.add_edge("process_jobs", "process_resumes")
            workflow.add_edge("process_resumes", "match_jobs")
            workflow.add_edge("match_jobs", "finalize")
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _wrap_async_node(self, async_func):
        """Wrap async node for sync execution"""
        def wrapper(state):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(async_func(state))
            finally:
                loop.close()
        return wrapper
    
    def run(
        self,
        job_query: Optional[str] = None,
        job_location: Optional[str] = None,
        resume_files: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run the pipeline with given parameters"""
        
        # Update configuration
        if job_query:
            self.config.job_query = job_query
        if job_location:
            self.config.job_location = job_location
        if resume_files:
            self.config.resume_files = resume_files
        
        # Apply additional overrides
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Start server if needed
        needs_llm = self.config.use_llm_matching
        if needs_llm and self.config.auto_start_server and self.server_manager:
            if not self.server_manager.ensure_running():
                logger.warning("LLM server not available, continuing without LLM")
                self.config.use_llm_matching = False
        
        # Initialize database if needed
        database = None
        if self.config.mode in [PipelineMode.PROCESS_RESUMES, PipelineMode.PROCESS_JOBS, 
                                PipelineMode.FULL_PIPELINE, PipelineMode.MATCH_ONLY]:
            from graph.database import graphDB
            database = graphDB()
            logger.info("Initialized database connection")
        
        # Initialize state
        initial_state = {
            "config": self.config,
            "start_time": datetime.now(),
            "errors": [],
            "diagnostics": {},
            "database": database
        }
        
        # Add resume files to state if processing resumes
        if self.config.mode in [PipelineMode.PROCESS_RESUMES, PipelineMode.FULL_PIPELINE]:
            if self.config.resume_path:
                # Single resume file specified
                initial_state["resume_files"] = [self.config.resume_path]
                logger.info(f"Processing single resume: {self.config.resume_path}")
            elif self.config.resume_files:
                # Multiple resume files specified
                initial_state["resume_files"] = self.config.resume_files
                logger.info(f"Processing {len(self.config.resume_files)} resume files")
            else:
                # No specific files, will process directory
                logger.info(f"Will process resumes from directory: {self.config.resume_directory}")
        
        # Add job search parameters if crawling jobs
        if self.config.mode in [PipelineMode.PROCESS_JOBS, PipelineMode.FULL_PIPELINE]:
            initial_state["job_query"] = self.config.job_query
            initial_state["job_location"] = self.config.job_location
            initial_state["max_jobs"] = self.config.max_jobs
        
        logger.info(f"Starting pipeline in mode: {self.config.mode.value}")
        
        # Run the graph
        try:
            final_state = self.graph.invoke(initial_state)
            
            # Stop server if configured
            if needs_llm and self.config.stop_server_on_complete and self.server_manager:
                self.server_manager.stop()
            
            return self._prepare_output(final_state)
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            
            if needs_llm and self.config.stop_server_on_complete and self.server_manager:
                self.server_manager.stop()
            
            return {
                "success": False,
                "error": str(e),
                "config": self.config.to_dict()
            }
    
    def _prepare_output(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final output"""
        
        return {
            "success": len(state.get("errors", [])) == 0,
            "timestamp": datetime.now().isoformat(),
            "config": state["config"].to_dict(),
            "results": {
                "jobs": state.get("processed_jobs", []),
                "resumes": state.get("processed_resumes", []),
                "matches": state.get("best_matches", [])[:20],
                "statistics": {
                    "jobs_processed": len(state.get("processed_jobs", [])),
                    "resumes_processed": len(state.get("processed_resumes", [])),
                    "matches_found": len(state.get("best_matches", [])),
                    "processing_time": state.get("diagnostics", {}).get("total_duration", 0)
                }
            },
            "errors": state.get("errors", []),
            "diagnostics": state.get("diagnostics", {})
        }


# -------------------------
# API Functions
# -------------------------

def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    """Main API function for running the pipeline"""
    pipeline = JobMatchingPipeline(config)
    return pipeline.run()


def run_job_processing(query: str, location: str = "remote", max_jobs: int = 10) -> Dict[str, Any]:
    """Crawl and process jobs"""
    config = {
        "mode": "process_jobs",
        "job_query": query,
        "job_location": location,
        "max_jobs": max_jobs
    }
    return run_pipeline(config)


def run_resume_processing(resume_files: List[str]) -> Dict[str, Any]:
    """Process resumes only"""
    config = {
        "mode": "process_resumes",
        "resume_files": resume_files
    }
    return run_pipeline(config)


def run_matching() -> Dict[str, Any]:
    """Run matching with existing data"""
    config = {
        "mode": "match_only"
    }
    return run_pipeline(config)


def run_full_pipeline(
    query: str,
    location: str = "remote",
    resume_directory: str = "data/resumes",
    max_jobs: int = 20
) -> Dict[str, Any]:
    """Run complete pipeline"""
    config = {
        "mode": "full_pipeline",
        "job_query": query,
        "job_location": location,
        "resume_directory": resume_directory,
        "max_jobs": max_jobs
    }
    return run_pipeline(config)


# -------------------------
# CLI Interface
# -------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Simplified Job Matching Pipeline")
    parser.add_argument("--mode", type=str, default="full_pipeline",
                       choices=["process_jobs", "process_resumes", "match_only", "full_pipeline"],
                       help="Pipeline execution mode")
    parser.add_argument("--query", type=str, default="python developer",
                       help="Job search query")
    parser.add_argument("--location", type=str, default="remote",
                       help="Job location")
    parser.add_argument("--max-jobs", type=int, default=20,
                       help="Maximum number of jobs to crawl (max: 50)")
    parser.add_argument("--resume-dir", type=str, default="data/resumes",
                       help="Directory containing resumes")
    parser.add_argument("--resume-path", type=str, default=None,
                       help="Path to a single resume file to process")
    parser.add_argument("--output-dir", type=str, default="data/pipeline_output",
                       help="Output directory")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM matching")
    parser.add_argument("--no-auto-server", action="store_true",
                       help="Disable automatic LLM server startup")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        "mode": args.mode,
        "job_query": args.query,
        "job_location": args.location,
        "max_jobs": args.max_jobs,
        "resume_directory": args.resume_dir,
        "resume_path": args.resume_path,  # Add single resume path
        "output_directory": args.output_dir,
        "use_llm_matching": not args.no_llm,
        "auto_start_server": not args.no_auto_server,  # Add auto server control
        "verbose": args.verbose,
        # For match_only mode, clear all matches. For full_pipeline, preserve existing
        "clear_all_matches": args.mode == "match_only"
    }
    
    print("\n" + "=" * 60)
    print("SIMPLIFIED JOB MATCHING PIPELINE")
    print("=" * 60)
    print(f"Mode: {args.mode}")
    if args.mode in ["process_jobs", "full_pipeline"]:
        print(f"Query: {args.query}")
        print(f"Location: {args.location}")
    if args.mode in ["process_resumes", "full_pipeline"]:
        if args.resume_path:
            print(f"Resume file: {args.resume_path}")
        else:
            print(f"Resume directory: {args.resume_dir}")
    print("=" * 60)
    
    # Run pipeline
    result = run_pipeline(config)
    
    # Display results
    if result.get("success"):
        print(f"\nPipeline completed successfully!")
        if "results" in result:
            stats = result["results"]["statistics"]
            if stats.get("jobs_processed"):
                print(f"Jobs: {stats['jobs_processed']}")
            if stats.get("resumes_processed"):
                print(f"Resumes: {stats['resumes_processed']}")
            if stats.get("matches_found"):
                print(f"Matches: {stats['matches_found']}")
            if stats.get("processing_time"):
                print(f"Time: {stats['processing_time']:.2f}s")
    else:
        print(f"\nPipeline failed: {result.get('error', 'Unknown error')}")

"""
Streaming matching module for incremental job-resume matching
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

from graph.database import graphDB
from graph.nodes.enhanced_matching import EnhancedMatcher
from graph.embeddings import JobEmbeddings
from graph.llm_server import LLMServerManager

logger = logging.getLogger(__name__)


class StreamingMatcher:
    """Handles incremental matching of jobs and resumes"""
    
    def __init__(self, emit_callback=None, auto_start_llm: bool = False):
        """
        Initialize streaming matcher
        
        Args:
            emit_callback: Function to call with status updates
            auto_start_llm: Whether to auto-start the LLM server if needed (default False for subprocess)
        """
        self.emit_callback = emit_callback or self._default_emit
        self.db = graphDB()
        
        # When running as subprocess from app.py, the LLM server is already running
        # Don't create a new one - just check if it's available
        self.server_manager = None
        if auto_start_llm:
            # Only create a new server if explicitly requested (for standalone mode)
            self.server_manager = LLMServerManager()
            if not self.server_manager.ensure_running():
                logger.warning("LLM server not available, continuing without LLM")
                auto_start_llm = False
        else:
            # Check if existing server is running (expected when run from app.py)
            try:
                import requests
                response = requests.get('http://localhost:8000/v1/models', timeout=2)
                if response.status_code == 200:
                    logger.info("Connected to existing LLM server at http://localhost:8000/v1")
                    self.emit("Connected to LLM server for enhanced matching")
                else:
                    logger.warning("LLM server not responding properly, LLM features may be unavailable")
                    self.emit("⚠ LLM server not available - matching without LLM analysis")
            except Exception as e:
                logger.warning(f"Could not connect to LLM server: {e}")
                self.emit("⚠ Could not connect to LLM server - matching without LLM analysis")
        
        # Pass config to ensure all components use the same LLM settings
        import os
        llm_config = {
            'llm_base_url': os.getenv("OPENAI_BASE_URL", 'http://localhost:8000/v1'),
            'llm_model': os.getenv("LLM_MODEL", 'qwen3-4b-instruct-2507-f16')
        }
        
        self.matcher = EnhancedMatcher(config=llm_config)
        self.embeddings = JobEmbeddings()
        
    def _default_emit(self, message: str):
        """Default emit function - just print"""
        print(message)
        
    def emit(self, message: str):
        """Emit a status message"""
        try:
            if self.emit_callback:
                self.emit_callback(message)
        except Exception as e:
            logger.error(f"Error emitting message: {e}")
    
    async def match_incrementally(
        self,
        resume_filter: Optional[str] = None,
        job_filter: Optional[str] = None,
        min_score: float = 0.3
    ):
        """
        Match jobs and resumes incrementally, emitting results as they're found
        
        Args:
            resume_filter: Optional filter for resume IDs
            job_filter: Optional filter for job IDs
            min_score: Minimum match score threshold
        """
        try:
            # Get all jobs and resumes
            jobs = self.db.get_all_jobs()
            resumes = self.db.get_all_resumes()
            
            if not jobs or not resumes:
                self.emit("No jobs or resumes found for matching")
                return
            
            self.emit(f"Starting incremental matching: {len(jobs)} jobs × {len(resumes)} resumes")
            
            # Apply filters if provided
            if resume_filter:
                resumes = [r for r in resumes if resume_filter in str(r.get('resume_id', ''))]
            if job_filter:
                jobs = [j for j in jobs if job_filter in str(j.get('job_id', ''))]
            
            total_combinations = len(jobs) * len(resumes)
            processed = 0
            matches_found = 0
            
            # Process each job-resume pair
            for job in jobs:
                job_id = job.get('job_id')
                job_title = job.get('title', 'Unknown Position')
                company = job.get('company', 'Unknown Company')
                
                self.emit(f"\nProcessing job: {job_title} at {company}")
                
                # Get job template
                job_template = json.loads(job.get('vocation_template', '{}')) if isinstance(job.get('vocation_template'), str) else job.get('vocation_template', {})
                if not job_template:
                    job_template = {
                        'title': job_title,
                        'technical_skills': job.get('skills', []),
                        'metadata': {'company': company}
                    }
                
                job_matches = []
                
                for resume in resumes:
                    resume_id = resume.get('resume_id')
                    resume_name = resume.get('name', 'Unknown')
                    
                    # Get resume template
                    resume_template = json.loads(resume.get('matching_template', '{}')) if isinstance(resume.get('matching_template'), str) else resume.get('matching_template', {})
                    if not resume_template:
                        resume_template = json.loads(resume.get('vocation_template', '{}')) if isinstance(resume.get('vocation_template'), str) else resume.get('vocation_template', {})
                    
                    if not resume_template:
                        continue
                    
                    try:
                        # Log that we're starting LLM analysis (only in debug mode)
                        logger.debug(f"Starting enhanced matching with LLM for {resume_name} -> {job_title}")
                        
                        # Emit clean status message
                        self.emit(f"  Analyzing: {resume_name}...")
                        
                        # Use _match_single which includes LLM assessment
                        detailed_match = self.matcher._match_single(
                            resume=resume,
                            job=job
                        )
                        
                        processed += 1
                        
                        # Log LLM results if available (only in debug mode)
                        if detailed_match and detailed_match.llm_score:
                            logger.debug(f"LLM assessment completed: score={detailed_match.llm_score:.2f}")
                            self.emit(f"  → LLM Analysis: {detailed_match.llm_score:.0%}")
                        
                        if detailed_match and detailed_match.overall_score >= min_score:
                            match_data = {
                                'job_id': job_id,
                                'resume_id': resume_id,
                                'overall_score': detailed_match.overall_score,
                                'title_match_score': detailed_match.title_alignment,  # Added title match score
                                'skills_score': detailed_match.skills_match,
                                'experience_score': detailed_match.experience_match,
                                'education_score': detailed_match.education_match,
                                'location_score': detailed_match.location_match,
                                'salary_score': detailed_match.salary_alignment,
                                'semantic_score': detailed_match.summary_to_description,  # Added semantic score
                                'llm_score': detailed_match.llm_score,
                                'llm_reasoning': detailed_match.llm_reasoning,
                                'llm_recommendations': detailed_match.llm_recommendations,
                                'skills_matched': detailed_match.matched_skills,
                                'skills_gap': detailed_match.missing_skills,  # Database expects skills_gap not missing_skills
                                'requirements_matched': detailed_match.matched_requirements if hasattr(detailed_match, 'matched_requirements') else [],
                                'requirements_gap': detailed_match.missing_requirements if hasattr(detailed_match, 'missing_requirements') else [],
                                'location_preference_met': detailed_match.location_preference_met,
                                'remote_preference_met': detailed_match.remote_preference_met,
                                'salary_match': detailed_match.salary_match,
                                'education_gaps': detailed_match.education_gaps,
                                'match_timestamp': datetime.now().isoformat()
                            }
                            
                            job_matches.append(match_data)
                            matches_found += 1
                            
                            # Emit match found immediately with LLM score if available
                            llm_info = f" [LLM: {detailed_match.llm_score:.0%}]" if detailed_match.llm_score else ""
                            self.emit(f"  Match found: {resume_name} (Score: {detailed_match.overall_score:.0%}){llm_info}")
                    
                    except Exception as e:
                        logger.error(f"Error matching job {job_id} with resume {resume_id}: {e}")
                    
                    # Emit progress periodically
                    if processed % 10 == 0:
                        progress = (processed / total_combinations) * 100
                        self.emit(f"Progress: {progress:.1f}% ({processed}/{total_combinations})")
                
                # Save matches for this job
                if job_matches:
                    saved_count = self.db.save_matches(job_matches)
                    self.emit(f"Saved {len(job_matches)} matches")
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            self.emit(f"\nMatching complete: Found {matches_found} matches from {processed} combinations")
            
        except Exception as e:
            logger.error(f"Error in incremental matching: {e}")
            self.emit(f"Error: {str(e)}")


async def main():
    """Main entry point for streaming matching"""
    import argparse
    import sys
    
    # Configure logging to show INFO level messages only in standalone mode
    # When run from app.py, we want clean output only
    
    parser = argparse.ArgumentParser(description='Incremental job-resume matching')
    parser.add_argument('--incremental', action='store_true', help='Run incremental matching')
    parser.add_argument('--resume-filter', help='Filter resumes by ID')
    parser.add_argument('--job-filter', help='Filter jobs by ID')
    parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    parser.add_argument('--standalone', action='store_true', help='Run in standalone mode (start own LLM server)')
    parser.add_argument('--verbose', action='store_true', help='Show detailed logging output')
    
    args = parser.parse_args()
    
    # Only configure logging if verbose or standalone mode
    if args.verbose or args.standalone:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    else:
        # Suppress most logging when run from app - only show critical errors
        logging.basicConfig(
            level=logging.ERROR,
            format='%(message)s',
            handlers=[logging.StreamHandler(sys.stderr)]
        )
    
    # Flush output for real-time updates
    def emit_to_flask(message: str):
        print(message)
        sys.stdout.flush()
    
    # Only auto-start LLM server if running in standalone mode
    matcher = StreamingMatcher(emit_callback=emit_to_flask, auto_start_llm=args.standalone)
    
    try:
        await matcher.match_incrementally(
            resume_filter=args.resume_filter,
            job_filter=args.job_filter,
            min_score=args.min_score
        )
    except KeyboardInterrupt:
        print("\nMatching interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Matching error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

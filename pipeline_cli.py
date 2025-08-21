#!/usr/bin/env python
"""
Standalone CLI for running pipeline nodes individually or in combinations
Each node can be run separately and data is persisted in LanceDB
"""

import os
import sys
import json
import logging
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.database import graphDB
from graph.crawler import BulkJobsCrawler
from graph.llm_server import LLMServerManager
from graph.settings import get_settings
from graph.cache_manager import CacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# -------------------------
# Standalone Node Functions
# -------------------------

async def crawl_and_process_jobs(
    query: str,
    location: str = "remote",
    max_jobs: int = 10,
    headless: bool = True,
    save_to_db: bool = True,
    use_llm: bool = True,
    output_dir: str = "data/pipeline_output"
) -> Dict[str, Any]:
    """
    Crawl and process jobs - always do both together
    Returns crawled and processed job data, saves to database
    """
    logger.info(f"Crawling and processing jobs: query='{query}', location='{location}', max={max_jobs}")
    
    result = {
        "success": False,
        "jobs": [],
        "processed_jobs": [],
        "errors": [],
        "stats": {}
    }
    
    try:
        # Step 1: Crawl jobs
        crawler = BulkJobsCrawler(
            headless=headless,
            data_dir=f"{output_dir}/jobs"
        )
        
        crawl_result = await crawler.get_all_jobs_expanded(
            query=query,
            location=location,
            max_jobs=max_jobs
        )
        
        if "error" in crawl_result:
            result["errors"].append(crawl_result["error"])
            return result
        
        jobs = crawl_result.get("jobs", [])
        result["jobs"] = jobs
        result["stats"]["crawl"] = {
            "total_found": crawl_result.get("total_jobs_found", 0),
            "processed": crawl_result.get("jobs_processed", 0),
            "captured": len(jobs)
        }
        
        # Step 2: Process the jobs to extract structured information
        logger.info(f"Processing {len(jobs)} crawled jobs")
        
        from graph.skill_extractor import extract_skills_from_text
        
        processed_jobs = []
        for job in jobs:
            try:
                job_text = job.get('text', '')
                
                # Extract skills
                skills = extract_skills_from_text(job_text)
                
                # Create universal template
                vocation_template = {
                    'source_type': 'job_posting',
                    'source_id': job.get('job_id', f"job_{datetime.now().timestamp()}"),
                    'title': job.get('title', 'Unknown Position'),
                    'company': job.get('company', 'Unknown Company'),
                    'location': job.get('location', location),
                    'technical_skills': skills,
                    'full_text': job_text,
                    'text_length': len(job_text),
                    'processed_at': datetime.now().isoformat()
                }
                
                job['vocation_template'] = vocation_template
                job['skills_extracted'] = len(skills)
                processed_jobs.append(job)
                
            except Exception as e:
                logger.error(f"Error processing job: {e}")
                processed_jobs.append(job)
        
        result["processed_jobs"] = processed_jobs
        result["stats"]["process"] = {
            "processed": len(processed_jobs),
            "with_skills": sum(1 for j in processed_jobs if j.get('skills_extracted', 0) > 0)
        }
        
        # Save to database if requested
        if save_to_db and processed_jobs:
            db = graphDB()
            count = db.add_jobs(processed_jobs, search_query=query, search_location=location)
            logger.info(f"Saved {count} processed jobs to database")
            result["stats"]["saved_to_db"] = count
        
        # Save to file
        output_file = Path(output_dir) / "jobs" / f"processed_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "jobs": processed_jobs,
                "stats": result["stats"]
            }, f, indent=2)
        
        result["success"] = True
        result["output_file"] = str(output_file)
        
    except Exception as e:
        logger.error(f"Job crawling/processing failed: {e}")
        result["errors"].append(str(e))
    
    return result


def process_resumes_standalone(
    resume_dir: str = "data/resumes",
    resume_files: Optional[List[str]] = None,
    save_to_db: bool = True,
    use_llm: bool = True,
    output_dir: str = "data/pipeline_output"
) -> Dict[str, Any]:
    """
    Standalone resume processing function
    Processes resumes and optionally saves to database
    """
    logger.info(f"Processing resumes from: {resume_dir}")
    
    result = {
        "success": False,
        "resumes": [],
        "errors": [],
        "stats": {}
    }
    
    try:
        # Get resume files
        if resume_files:
            files = resume_files
        else:
            resume_path = Path(resume_dir)
            files = list(resume_path.glob("*.txt")) + \
                   list(resume_path.glob("*.pdf")) + \
                   list(resume_path.glob("*.docx"))
            files = [str(f) for f in files]
        
        if not files:
            logger.warning("No resume files found")
            result["errors"].append("No resume files found")
            return result
        
        logger.info(f"Found {len(files)} resume files")
        
        # Process resumes
        if use_llm:
            # Ensure LLM server is running
            server_manager = LLMServerManager()
            if not server_manager.ensure_running():
                logger.warning("LLM server not available, using basic extraction")
                use_llm = False
        
        processed_resumes = []
        
        for file_path in files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
                
                # Extract skills using simple extractor
                from graph.skill_extractor import extract_skills_from_text
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
                    "summary": text[:500],  # First 500 chars as summary
                    "vocation_template": {
                        "source_type": "resume",
                        "candidate_profile": {
                            "name": _extract_name_from_text(text),
                            "email": _extract_email_from_text(text),
                            "phone": _extract_phone_from_text(text),
                            "location": "",
                            "summary": text[:500]
                        },
                        "requirements_match": {
                            "technical_skills": skills
                        },
                        "skills": skills,
                        "full_text": text
                    }
                }
                
                processed_resumes.append(resume_data)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
                result["errors"].append(f"Failed to process {Path(file_path).name}: {str(e)}")
        
        result["resumes"] = processed_resumes
        result["stats"]["processed"] = len(processed_resumes)
        
        # Save to database if requested
        if save_to_db and processed_resumes:
            db = graphDB()
            count = db.add_resumes(processed_resumes)
            logger.info(f"Saved {count} resumes to database")
            result["stats"]["saved_to_db"] = count
        
        # Save to file
        output_file = Path(output_dir) / "resumes" / f"processed_resumes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "resumes": processed_resumes,
                "stats": result["stats"]
            }, f, indent=2)
        
        result["success"] = True
        result["output_file"] = str(output_file)
        
    except Exception as e:
        logger.error(f"Resume processing failed: {e}")
        result["errors"].append(str(e))
    
    return result


def match_jobs_standalone(
    load_from_db: bool = True,
    job_files: Optional[List[str]] = None,
    resume_files: Optional[List[str]] = None,
    use_llm: bool = True,
    top_k: int = 10,
    min_score: float = 0.3,
    output_dir: str = "data/pipeline_output"
) -> Dict[str, Any]:
    """
    Standalone matching function
    Loads jobs and resumes from database or files and performs matching
    """
    logger.info("Running standalone matching")
    
    result = {
        "success": False,
        "matches": [],
        "errors": [],
        "stats": {}
    }
    
    try:
        jobs = []
        resumes = []
        
        # Load data from database
        if load_from_db:
            db = graphDB()
            jobs = db.get_all_jobs()
            resumes = db.get_all_resumes()
            logger.info(f"Loaded {len(jobs)} jobs and {len(resumes)} resumes from database")
        
        # Load from files if specified
        if job_files:
            for file_path in job_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if "jobs" in data:
                        jobs.extend(data["jobs"])
                    elif isinstance(data, list):
                        jobs.extend(data)
        
        if resume_files:
            for file_path in resume_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if "resumes" in data:
                        resumes.extend(data["resumes"])
                    elif isinstance(data, list):
                        resumes.extend(data)
        
        if not jobs or not resumes:
            result["errors"].append(f"Insufficient data: {len(jobs)} jobs, {len(resumes)} resumes")
            return result
        
        logger.info(f"Matching {len(resumes)} resumes against {len(jobs)} jobs")
        
        # Ensure LLM server if needed
        if use_llm:
            server_manager = LLMServerManager()
            if not server_manager.ensure_running():
                logger.warning("LLM server not available, using basic matching")
                use_llm = False
        
        # Perform matching using the proper pipeline
        from graph.nodes.job_matching import JobMatchingPipeline, JobMatchConfig
        
        # Configure matching
        config = JobMatchConfig(
            min_score=min_score,
            top_k=top_k,
            semantic_weight=0.4,
            skills_weight=0.3,
            experience_weight=0.2,
            llm_weight=0.1 if use_llm else 0.0
        )
        
        # Initialize pipeline
        pipeline = JobMatchingPipeline(config=config)
        
        all_matches = []
        
        for resume in resumes:
            try:
                # Parse universal template if it's a string
                resume_template = resume.get('vocation_template')
                if isinstance(resume_template, str):
                    try:
                        resume_template = json.loads(resume_template)
                    except:
                        resume_template = None
                
                if not resume_template:
                    # If no template, create a basic one
                    resume_template = {
                        'candidate_profile': {
                            'name': resume.get('name', 'Unknown'),
                            'email': resume.get('email', ''),
                            'years_experience': None
                        },
                        'requirements_match': {
                            'technical_skills': []
                        },
                        'work_history': [],
                        'full_text': resume.get('full_text', '')
                    }
                
                # The universal template has skills under 'skills' not 'requirements_match'
                if 'skills' in resume_template and not resume_template.get('requirements_match'):
                    resume_template['requirements_match'] = {
                        'technical_skills': []
                    }
                    
                    # Extract skill names from the skills list
                    skills = resume_template['skills']
                    if skills and isinstance(skills, list):
                        skill_names = []
                        for skill in skills:
                            if isinstance(skill, dict):
                                skill_name = skill.get('skill_name', skill.get('name', skill.get('skill', '')))
                                if skill_name:
                                    skill_names.append(skill_name)
                            elif isinstance(skill, str):
                                skill_names.append(skill)
                        resume_template['requirements_match']['technical_skills'] = skill_names
                
                # Also ensure we have the full_text field
                if 'full_text' not in resume_template:
                    resume_template['full_text'] = resume.get('full_text', '')
                
                # Use the pipeline to match
                result_obj = pipeline.match_resume_to_jobs(
                    resume_template,
                    jobs,
                    use_llm=use_llm,
                    verbose=False
                )
                
                # Convert matches to the expected format
                for match in result_obj.matches:
                    match_record = {
                        'resume_id': resume.get('resume_id', ''),
                        'resume_name': result_obj.candidate_name,
                        'job_id': match.job_id,
                        'job_title': match.job_title,
                        'company': match.company,
                        'overall_score': match.overall_score,
                        'semantic_score': match.semantic_score,
                        'skills_score': match.skills_score,
                        'experience_score': match.experience_score,
                        'llm_score': match.llm_score,
                        'match_reasons': match.match_reasons,
                        'skills_gap': match.missing_skills,
                        'llm_assessment': match.llm_assessment
                    }
                    all_matches.append(match_record)
                    
            except Exception as e:
                logger.error(f"Error matching resume: {e}")
        
        # Sort matches by score
        all_matches.sort(key=lambda x: x['overall_score'], reverse=True)
        
        # Keep top k matches
        top_matches = all_matches[:top_k]
        
        result["matches"] = top_matches
        result["stats"] = {
            "total_matches": len(all_matches),
            "top_matches": len(top_matches),
            "jobs_processed": len(jobs),
            "resumes_processed": len(resumes)
        }
        
        # Save matches to database
        if load_from_db and top_matches:
            db = graphDB()
            pipeline_run_id = f"standalone_{datetime.now().isoformat()}"
            count = db.save_matches(top_matches, pipeline_run_id=pipeline_run_id)
            logger.info(f"Saved {count} matches to database")
            result["stats"]["saved_to_db"] = count
        
        # Save to file
        output_file = Path(output_dir) / "matches" / f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "matches": top_matches,
                "stats": result["stats"]
            }, f, indent=2)
        
        result["success"] = True
        result["output_file"] = str(output_file)
        
    except Exception as e:
        logger.error(f"Matching failed: {e}")
        result["errors"].append(str(e))
    
    return result


def process_jobs_standalone(
    load_from_db: bool = True,
    job_files: Optional[List[str]] = None,
    use_llm: bool = True,
    output_dir: str = "data/pipeline_output"
) -> Dict[str, Any]:
    """
    Standalone job processing function
    Processes raw jobs to extract structured information
    """
    logger.info("Processing jobs")
    
    result = {
        "success": False,
        "jobs": [],
        "errors": [],
        "stats": {}
    }
    
    try:
        jobs = []
        
        # Load from database
        if load_from_db:
            db = graphDB()
            jobs = db.get_all_jobs()
            logger.info(f"Loaded {len(jobs)} jobs from database")
        
        # Load from files if specified
        if job_files:
            for file_path in job_files:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    if "jobs" in data:
                        jobs.extend(data["jobs"])
                    elif isinstance(data, list):
                        jobs.extend(data)
        
        if not jobs:
            result["errors"].append("No jobs to process")
            return result
        
        logger.info(f"Processing {len(jobs)} jobs")
        
        # Process each job
        from graph.skill_extractor import extract_skills_from_text
        
        processed_jobs = []
        for job in jobs:
            try:
                job_text = job.get('text', '')
                
                # Extract skills
                skills = extract_skills_from_text(job_text)
                
                # Create universal template
                vocation_template = {
                    'source_type': 'job_posting',
                    'source_id': job.get('job_id', f"job_{datetime.now().timestamp()}"),
                    'title': job.get('title', 'Unknown Position'),
                    'company': job.get('company', 'Unknown Company'),
                    'location': job.get('location', ''),
                    'technical_skills': skills,
                    'full_text': job_text,
                    'text_length': len(job_text),
                    'processed_at': datetime.now().isoformat()
                }
                
                job['vocation_template'] = vocation_template
                job['skills_extracted'] = len(skills)
                processed_jobs.append(job)
                
            except Exception as e:
                logger.error(f"Error processing job: {e}")
                processed_jobs.append(job)
        
        result["jobs"] = processed_jobs
        result["stats"]["processed"] = len(processed_jobs)
        
        # Save to file
        output_file = Path(output_dir) / "jobs" / f"processed_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump({
                "jobs": processed_jobs,
                "stats": result["stats"]
            }, f, indent=2)
        
        result["success"] = True
        result["output_file"] = str(output_file)
        
    except Exception as e:
        logger.error(f"Job processing failed: {e}")
        result["errors"].append(str(e))
    
    return result


# -------------------------
# Helper Functions
# -------------------------

def _extract_name_from_text(text: str) -> str:
    """Extract name from resume text (basic implementation)"""
    lines = text.split('\n')
    for line in lines[:5]:  # Check first 5 lines
        line = line.strip()
        if line and len(line.split()) <= 4 and not any(char.isdigit() for char in line):
            # Likely a name if it's short and has no numbers
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
# Combined Pipeline Functions
# -------------------------


async def full_pipeline(
    query: str,
    location: str = "remote",
    max_jobs: int = 10,
    resume_dir: str = "data/resumes",
    headless: bool = True,
    use_llm: bool = True,
    top_k: int = 10,
    output_dir: str = "data/pipeline_output"
) -> Dict[str, Any]:
    """
    Run full pipeline: crawl+process jobs, process resumes, match
    """
    logger.info("Running full pipeline")
    
    results = {}
    
    # Step 1: Crawl and process jobs (always together)
    jobs_result = await crawl_and_process_jobs(
        query=query,
        location=location,
        max_jobs=max_jobs,
        headless=headless,
        save_to_db=True,
        use_llm=use_llm,
        output_dir=output_dir
    )
    results["jobs"] = jobs_result
    
    if not jobs_result["success"]:
        return {
            "success": False,
            "error": "Job crawling/processing failed",
            "results": results
        }
    
    # Step 2: Process resumes
    resume_result = process_resumes_standalone(
        resume_dir=resume_dir,
        save_to_db=True,
        use_llm=use_llm,
        output_dir=output_dir
    )
    results["resumes"] = resume_result
    
    if not resume_result["success"]:
        return {
            "success": False,
            "error": "Resume processing failed",
            "results": results
        }
    
    # Step 3: Match
    match_result = match_jobs_standalone(
        load_from_db=True,
        use_llm=use_llm,
        top_k=top_k,
        output_dir=output_dir
    )
    results["matches"] = match_result
    
    return {
        "success": match_result["success"],
        "results": results,
        "summary": {
            "jobs_crawled": jobs_result["stats"]["crawl"].get("captured", 0),
            "jobs_processed": jobs_result["stats"]["process"].get("processed", 0),
            "resumes_processed": resume_result["stats"].get("processed", 0),
            "matches_found": match_result["stats"].get("total_matches", 0),
            "top_matches": match_result["stats"].get("top_matches", 0)
        }
    }


# -------------------------
# Main CLI
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Job Matching Pipeline CLI")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Crawl command (always includes processing)
    crawl_parser = subparsers.add_parser('crawl', help='Crawl and process jobs')
    crawl_parser.add_argument('--query', type=str, default='python developer', help='Job search query')
    crawl_parser.add_argument('--location', type=str, default='remote', help='Job location')
    crawl_parser.add_argument('--max-jobs', type=int, default=50, help='Maximum jobs to crawl')
    crawl_parser.add_argument('--no-headless', action='store_true', help='Show browser')
    crawl_parser.add_argument('--no-db', action='store_true', help='Skip database save')
    crawl_parser.add_argument('--no-llm', action='store_true', help='Skip LLM processing')
    
    # Process resumes command
    resume_parser = subparsers.add_parser('process-resumes', help='Process resumes only')
    resume_parser.add_argument('--dir', type=str, default='data/resumes', help='Resume directory')
    resume_parser.add_argument('--files', nargs='+', help='Specific resume files')
    resume_parser.add_argument('--no-llm', action='store_true', help='Skip LLM processing')
    resume_parser.add_argument('--no-db', action='store_true', help='Skip database save')
    
    # Reprocess jobs command (for existing job data)
    reprocess_jobs_parser = subparsers.add_parser('reprocess-jobs', help='Reprocess existing job data')
    reprocess_jobs_parser.add_argument('--files', nargs='+', help='Job files to process')
    reprocess_jobs_parser.add_argument('--no-llm', action='store_true', help='Skip LLM processing')
    reprocess_jobs_parser.add_argument('--from-db', action='store_true', help='Load jobs from database')
    
    # Match command
    match_parser = subparsers.add_parser('match', help='Match resumes to jobs')
    match_parser.add_argument('--job-files', nargs='+', help='Job files')
    match_parser.add_argument('--resume-files', nargs='+', help='Resume files')
    match_parser.add_argument('--from-db', action='store_true', default=True, help='Load from database')
    match_parser.add_argument('--no-llm', action='store_true', help='Skip LLM matching')
    match_parser.add_argument('--top-k', type=int, default=1000, help='Top K matches')
    match_parser.add_argument('--min-score', type=float, default=0.3, help='Minimum match score')
    
    # Full pipeline command
    full_parser = subparsers.add_parser('full', help='Run full pipeline')
    full_parser.add_argument('--query', type=str, default='python developer', help='Job search query')
    full_parser.add_argument('--location', type=str, default='remote', help='Job location')
    full_parser.add_argument('--max-jobs', type=int, default=50, help='Maximum jobs to crawl')
    full_parser.add_argument('--resume-dir', type=str, default='data/resumes', help='Resume directory')
    full_parser.add_argument('--no-headless', action='store_true', help='Show browser')
    full_parser.add_argument('--no-llm', action='store_true', help='Skip LLM processing')
    full_parser.add_argument('--top-k', type=int, default=1000, help='Top K matches')
    
    # Common arguments
    parser.add_argument('--output-dir', type=str, default='data/pipeline_output', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Execute command
    if args.command == 'crawl':
        result = asyncio.run(crawl_and_process_jobs(
            query=args.query,
            location=args.location,
            max_jobs=args.max_jobs,
            headless=not args.no_headless,
            save_to_db=not args.no_db,
            use_llm=not args.no_llm,
            output_dir=args.output_dir
        ))
        
    elif args.command == 'process-resumes':
        result = process_resumes_standalone(
            resume_dir=args.dir,
            resume_files=args.files,
            save_to_db=not args.no_db,
            use_llm=not args.no_llm,
            output_dir=args.output_dir
        )
        
    elif args.command == 'reprocess-jobs':
        result = process_jobs_standalone(
            load_from_db=args.from_db,
            job_files=args.files,
            use_llm=not args.no_llm,
            output_dir=args.output_dir
        )
        
    elif args.command == 'match':
        result = match_jobs_standalone(
            load_from_db=args.from_db,
            job_files=args.job_files,
            resume_files=args.resume_files,
            use_llm=not args.no_llm,
            top_k=args.top_k,
            min_score=args.min_score,
            output_dir=args.output_dir
        )
        
    elif args.command == 'full':
        result = asyncio.run(full_pipeline(
            query=args.query,
            location=args.location,
            max_jobs=args.max_jobs,
            resume_dir=args.resume_dir,
            headless=not args.no_headless,
            use_llm=not args.no_llm,
            top_k=args.top_k,
            output_dir=args.output_dir
        ))
        
    else:
        parser.print_help()
        return
    
    # Print results
    print("\n" + "=" * 60)
    if result.get("success"):
        print("✅ SUCCESS")
    else:
        print("❌ FAILED")
    print("=" * 60)
    
    # Print statistics
    if "stats" in result:
        print("\nStatistics:")
        for key, value in result["stats"].items():
            print(f"  {key}: {value}")
    
    if "summary" in result:
        print("\nSummary:")
        for key, value in result["summary"].items():
            print(f"  {key}: {value}")
    
    # Print errors if any
    if result.get("errors"):
        print("\nErrors:")
        for error in result["errors"]:
            print(f"  - {error}")
    
    # Print output file if saved
    if "output_file" in result:
        print(f"\nOutput saved to: {result['output_file']}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()

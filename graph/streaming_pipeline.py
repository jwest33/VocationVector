"""
Streaming pipeline for real-time job processing and matching
Processes jobs individually and emits updates as they complete
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from graph.crawler import BulkJobsCrawler
from graph.database import graphDB
from graph.llm_server import LLMServerManager
from graph.settings import get_settings
from graph.embeddings import JobEmbeddings

# Import processing nodes
from graph.nodes.job_processing import JobLLMProcessor
from graph.nodes.job_matching import JobMatchingPipeline
from graph.nodes.enhanced_matching import EnhancedMatcher

logger = logging.getLogger(__name__)


class StreamingCrawler(BulkJobsCrawler):
    """Enhanced crawler that yields jobs one at a time"""
    
    async def crawl_jobs_incrementally(
        self, 
        query: str, 
        location: str = "",
        max_jobs: int = 10,
        on_job_found: Optional[Callable] = None
    ):
        """
        Crawl jobs and yield them one at a time as they're found
        
        Args:
            query: Job search query
            location: Location for the search
            max_jobs: Maximum number of jobs to crawl
            on_job_found: Callback when a job is found (gets raw job data)
        """
        browser = None
        try:
            from playwright.async_api import async_playwright
            from urllib.parse import quote_plus
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--window-size=1920,1080',
                    ]
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36',
                    locale='en-US',
                    permissions=[],
                    geolocation=None
                )
                
                page = await context.new_page()
                
                # Navigate to Google Jobs
                if 'jobs' not in query.lower():
                    query = f"{query} jobs"
                search_term = f"{query} {location}".strip()
                encoded_query = quote_plus(search_term)
                location_param = f"&near={quote_plus(location)}" if location else ""
                url = f"https://www.google.com/search?q={encoded_query}&hl=en{location_param}"
                
                logger.info(f"Searching for: {search_term}")
                await page.goto(url, wait_until='domcontentloaded')
                await asyncio.sleep(0.5)
                
                # Handle location prompts
                await page.keyboard.press('Escape')
                await asyncio.sleep(0.5)
                
                # Click Jobs tab
                try:
                    jobs_tab = page.locator("a:has-text('Jobs')").first
                    if await jobs_tab.is_visible(timeout=1000):
                        await jobs_tab.click()
                        await asyncio.sleep(1)
                except:
                    jobs_url = url.replace("/search?", "/search?ibp=htl;jobs&")
                    await page.goto(jobs_url, wait_until='domcontentloaded')
                    await asyncio.sleep(0.5)
                
                # Find job cards
                working_selector = 'div.EimVGf'
                cards = await page.query_selector_all(working_selector)
                
                if not cards:
                    for selector in ['li.iFjolb', 'div.PwjeAc', 'div[role="listitem"]']:
                        cards = await page.query_selector_all(selector)
                        if cards:
                            working_selector = selector
                            break
                
                if not cards:
                    logger.warning("No job cards found")
                    return
                
                # If we need more jobs than currently visible, try to load more
                initial_count = len(cards)
                if initial_count < max_jobs:
                    logger.info(f"Found {initial_count} jobs, attempting to load more (target: {max_jobs})")
                    
                    # Try scrolling to load more jobs
                    last_count = initial_count
                    attempts = 0
                    max_attempts = 5
                    
                    while len(cards) < max_jobs and attempts < max_attempts:
                        # Scroll to the last card
                        await page.evaluate('''
                            const cards = document.querySelectorAll('%s');
                            if (cards.length > 0) {
                                cards[cards.length - 1].scrollIntoView({ behavior: 'smooth', block: 'end' });
                            }
                        ''' % working_selector)
                        
                        await asyncio.sleep(1.5)  # Wait for new jobs to load
                        
                        # Check for "More jobs" button and click if found
                        try:
                            more_button = await page.query_selector('button:has-text("More jobs"), a:has-text("More jobs")')
                            if more_button and await more_button.is_visible():
                                await more_button.click()
                                await asyncio.sleep(1.5)
                        except:
                            pass
                        
                        # Re-query cards
                        cards = await page.query_selector_all(working_selector)
                        
                        if len(cards) == last_count:
                            # No new jobs loaded, stop trying
                            logger.info(f"No more jobs available (stuck at {len(cards)})")
                            break
                        
                        logger.info(f"Loaded {len(cards)} jobs so far")
                        last_count = len(cards)
                        attempts += 1
                
                jobs_to_process = min(len(cards), max_jobs)
                logger.info(f"Found {len(cards)} job cards, processing {jobs_to_process}")
                
                # Process each job individually and yield immediately
                for i in range(jobs_to_process):
                    try:
                        # Re-query cards to avoid stale references
                        current_cards = await page.query_selector_all(working_selector)
                        if i >= len(current_cards):
                            continue
                        
                        # Get preview text
                        card_text = await current_cards[i].text_content()
                        card_preview = (card_text[:60] + "...") if card_text and len(card_text) > 60 else card_text
                        
                        logger.info(f"Processing job {i+1}/{jobs_to_process}: {card_preview}")
                        
                        # Click the card
                        await current_cards[i].click()
                        await asyncio.sleep(0.5)  # Reduced wait time
                        
                        # Expand descriptions
                        try:
                            show_full_btn = page.locator("text='Show full description'").first
                            if await show_full_btn.is_visible(timeout=500):
                                await show_full_btn.click()
                                await asyncio.sleep(0.2)
                        except:
                            pass
                        
                        # Extract job data
                        job_data = await page.evaluate("""
                            () => {
                                const allDivs = document.querySelectorAll('div');
                                let bestMatch = '';
                                let bestDiv = null;
                                let largestSize = 0;
                                const applyLinks = [];
                                let postedDate = null;
                                
                                for (const div of allDivs) {
                                    const rect = div.getBoundingClientRect();
                                    if (rect.width === 0 || rect.height === 0) continue;
                                    
                                    const text = div.innerText || '';
                                    if (text.length < 500) continue;
                                    if (!text.includes('Apply')) continue;
                                    
                                    const hasJobContent = 
                                        (text.includes('Full-time') || text.includes('Part-time') || 
                                         text.includes('Contract') || text.includes('Remote')) &&
                                        text.includes('Apply');
                                    
                                    if (hasJobContent) {
                                        const isRightSide = rect.left > window.innerWidth / 3;
                                        if (isRightSide && text.length > largestSize) {
                                            largestSize = text.length;
                                            bestMatch = text;
                                            bestDiv = div;
                                        }
                                    }
                                }
                                
                                // Extract apply links and posted date from the best match div
                                if (bestDiv) {
                                    // Find all links that contain "Apply" or lead to application pages
                                    const links = bestDiv.querySelectorAll('a');
                                    for (const link of links) {
                                        const href = link.href;
                                        const text = (link.textContent || '').toLowerCase();
                                        
                                        // Check if this is an apply link
                                        if (href && (
                                            text.includes('apply') ||
                                            href.includes('apply') ||
                                            href.includes('career') ||
                                            href.includes('job') ||
                                            href.includes('workday') ||
                                            href.includes('greenhouse') ||
                                            href.includes('lever') ||
                                            href.includes('taleo') ||
                                            href.includes('recruiting')
                                        )) {
                                            applyLinks.push({
                                                url: href,
                                                text: link.textContent.trim()
                                            });
                                        }
                                    }
                                    
                                    // Try to extract posted date from the details panel
                                    const allText = bestDiv.innerText || '';
                                    
                                    // First look for "Date posted" label specifically
                                    const datePostedMatch = allText.match(/Date posted[:\s]*(\d+\+?\s*(day|week|month|hour|minute)s?\s*ago|just posted|today|yesterday)/i);
                                    if (datePostedMatch && datePostedMatch[1]) {
                                        postedDate = datePostedMatch[1].trim();
                                    } else {
                                        // Fallback to other date patterns
                                        const datePatterns = [
                                            /Posted[:\s]*(\d+\+?\s*(day|week|month|hour|minute)s?\s*ago)/gi,
                                            /(\d+\+?\s*(day|week|month|hour|minute)s?\s*ago)(?=\s*·|\s*\||\s*-|$)/gi,
                                            /(just posted|today|yesterday)/gi,
                                            /via\s+[^\s]+\s+(\d+\+?\s*(day|week|month)s?\s*ago)/gi
                                        ];
                                        
                                        for (const pattern of datePatterns) {
                                            const match = allText.match(pattern);
                                            if (match && match[0]) {
                                                // Clean up the match to get just the date part
                                                const cleanMatch = match[0].replace(/^(Posted|via\s+\S+)\s*/i, '').trim();
                                                postedDate = cleanMatch;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (bestMatch) {
                                    return {
                                        text: bestMatch,
                                        apply_links: applyLinks,
                                        posted_date: postedDate
                                    };
                                }
                                
                                return { text: '', apply_links: [], posted_date: null };
                            }
                        """)
                        
                        if job_data and job_data.get('text'):
                            job_info = {
                                'text': job_data.get('text'),
                                'job_index': i,
                                'apply_links': job_data.get('apply_links', []),
                                'posted_date': job_data.get('posted_date'),
                                'crawled_at': datetime.now().isoformat()
                            }
                            
                            # Emit the job immediately via callback
                            if on_job_found:
                                await on_job_found(job_info)
                            
                            yield job_info
                        
                    except Exception as e:
                        logger.error(f"Error processing job {i}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Crawler error: {e}")
        finally:
            if browser:
                await browser.close()


class StreamingPipeline:
    """Pipeline that processes jobs and matches incrementally"""
    
    def __init__(self, emit_callback: Optional[Callable] = None, auto_start_llm: bool = False):
        """
        Initialize streaming pipeline
        
        Args:
            emit_callback: Function to call with status updates
            auto_start_llm: Whether to auto-start the LLM server if needed (default False for subprocess)
        """
        self.emit_callback = emit_callback or self._default_emit
        self.db = graphDB()
        self.auto_start_llm = auto_start_llm
        
        # When running as subprocess from app.py, the LLM server is already running
        # Don't create a new one - just check if it's available
        self.server_manager = None
        if self.auto_start_llm:
            # Only create a new server if explicitly requested (for standalone mode)
            self.server_manager = LLMServerManager()
            if not self.server_manager.ensure_running():
                logger.warning("LLM server not available, continuing without LLM")
                self.auto_start_llm = False
        else:
            # Check if existing server is running (expected when run from app.py)
            import os
            # Get LLM server URL from environment or construct from host/port
            llm_host = os.getenv("LLM_SERVER_HOST", "localhost")
            llm_port = os.getenv("LLM_SERVER_PORT", "8000")
            base_url = os.getenv("OPENAI_BASE_URL", f"http://{llm_host}:{llm_port}/v1")
            
            try:
                import requests
                response = requests.get(f"{base_url}/models", timeout=2)
                if response.status_code == 200:
                    logger.info(f"Connected to existing LLM server at {base_url}")
                else:
                    logger.warning("LLM server not responding properly, LLM features may be unavailable")
            except Exception as e:
                logger.warning(f"Could not connect to LLM server: {e}")
        
        # Pass config to ensure all components use the same LLM settings
        llm_config = {
            'llm_base_url': base_url,
            'llm_model': os.getenv("LLM_MODEL", 'qwen3-4b-instruct-2507-f16')
        }
        
        self.job_processor = JobLLMProcessor()
        self.job_matcher = JobMatchingPipeline()
        self.enhanced_matcher = EnhancedMatcher(config=llm_config)
        self.embeddings = JobEmbeddings()
        
    def _default_emit(self, event_type: str, data: Dict):
        """Default emit function - just print"""
        logger.info(f"[{event_type}] {data}")
    
    def emit(self, event_type: str, data: Dict):
        """Emit an event"""
        try:
            if self.emit_callback:
                self.emit_callback(event_type, data)
        except Exception as e:
            logger.error(f"Error emitting event: {e}")
    
    async def process_job_incrementally(self, raw_job: Dict) -> Optional[Dict]:
        """
        Process a single job and save to database immediately
        
        Args:
            raw_job: Raw job data from crawler
            
        Returns:
            Processed job data or None if processing failed
        """
        try:
            # Emit crawled event
            self.emit('job_crawled', {
                'message': f"Found job {raw_job.get('job_index', 0) + 1}",
                'job_index': raw_job.get('job_index', 0)
            })
            
            # Extract structured data with LLM
            job_text = raw_job.get('text', '')
            if not job_text:
                return None
            
            # Process with LLM - using fixed extraction that doesn't add AWS to everything
            self.emit('job_processing', {
                'message': f"Processing job {raw_job.get('job_index', 0) + 1} with LLM",
                'job_index': raw_job.get('job_index', 0)
            })
            
            # extract_job_info is synchronous, not async
            extraction = self.job_processor.extract_job_info(job_text)
            # Convert extraction to dictionary format for compatibility
            template = extraction.model_dump() if hasattr(extraction, 'model_dump') else extraction.dict()
            
            if not template:
                logger.warning(f"Failed to process job {raw_job.get('job_index')}")
                return None
            
            # Build processed job object - mirroring the pipeline.py structure
            # Extract and format skills properly
            skills_list = []
            for skill in template.get('technical_skills', []):
                if isinstance(skill, dict):
                    skill_name = skill.get('skill_name', skill.get('skill', ''))
                    if skill_name:
                        skills_list.append(skill_name)
                elif isinstance(skill, str):
                    skills_list.append(skill)
            
            # Add tools/technologies to skills list
            for tool in template.get('tools_technologies', []):
                if isinstance(tool, str) and tool:
                    skills_list.append(tool)
            
            # Extract requirements from experience requirements
            requirements_list = []
            for exp_req in template.get('experience_requirements', []):
                if isinstance(exp_req, str):
                    requirements_list.append(exp_req)
                elif isinstance(exp_req, dict):
                    desc = exp_req.get('description', '')
                    if desc:
                        requirements_list.append(desc)
            
            # Add education requirements to requirements
            for edu_req in template.get('education_requirements', []):
                if isinstance(edu_req, dict):
                    degree = edu_req.get('degree_level', edu_req.get('degree', ''))
                    field = edu_req.get('field_of_study', edu_req.get('field', ''))
                    if degree:
                        req_text = f"{degree}"
                        if field:
                            req_text += f" in {field}"
                        requirements_list.append(req_text)
            
            # Extract benefits properly
            benefits_list = template.get('benefits', [])
            
            # Build a proper description from summary and key responsibilities
            description_parts = []
            if template.get('job_summary'):
                description_parts.append(template.get('job_summary'))
            elif template.get('summary'):
                description_parts.append(template.get('summary'))
            
            # Add key responsibilities to description if available
            responsibilities = template.get('responsibilities', [])
            if responsibilities and isinstance(responsibilities, list):
                description_parts.append("\n\nKey Responsibilities:")
                for resp in responsibilities[:5]:  # Limit to first 5
                    if resp:
                        description_parts.append(f"• {resp}")
            
            description = '\n'.join(description_parts) if description_parts else job_text[:500]
            
            # Build complete processed job matching pipeline.py structure
            processed_job = {
                'text': job_text,
                'job_index': raw_job.get('job_index', 0),
                'title': template.get('title', 'Unknown Position'),
                'company': template.get('company', 'Unknown Company'),
                'location': template.get('location', template.get('location_text', 'Not specified')),
                'description': description,
                'skills': skills_list,
                'requirements': requirements_list if requirements_list else ['No specific requirements listed'],
                'responsibilities': responsibilities if responsibilities else ['See job description'],
                'benefits': benefits_list if benefits_list else ['Benefits package available'],
                'salary_min': template.get('salary_min'),
                'salary_max': template.get('salary_max'),
                'employment_type': template.get('employment_type'),
                'remote_policy': template.get('remote_policy'),
                'vocation_template': template,  # Store the full template for advanced features
                'years_experience_required': template.get('years_experience_required'),
                'education_requirements': template.get('education_requirements', []),
                'apply_links': raw_job.get('apply_links', []),  # Include apply links
                'posted_date': raw_job.get('posted_date'),  # Include posted date
                'crawled_at': raw_job.get('crawled_at'),
                'processed_at': datetime.now().isoformat()
            }
            
            # Save to database immediately (will return existing ID if duplicate)
            job_id = self.db.add_job(processed_job)
            if job_id:
                processed_job['job_id'] = job_id
                
                # Check if this was a duplicate (job_id doesn't contain current timestamp)
                current_timestamp = datetime.now().isoformat()[:19]  # Match precision of job_id timestamp
                is_duplicate = current_timestamp not in job_id
                
                if is_duplicate:
                    self.emit('job_duplicate', {
                        'message': f"Duplicate job found: {processed_job['title']} at {processed_job['company']}",
                        'job': {
                            'job_id': job_id,
                            'title': processed_job['title'],
                            'company': processed_job['company'],
                            'location': processed_job['location']
                        }
                    })
                    # Skip matching for duplicates since they've already been matched
                    logger.info(f"Skipping duplicate job: {processed_job['title']} at {processed_job['company']}")
                else:
                    self.emit('job_saved', {
                        'message': f"Saved: {processed_job['title']} at {processed_job['company']}",
                        'job': {
                            'job_id': job_id,
                            'title': processed_job['title'],
                            'company': processed_job['company'],
                            'location': processed_job['location'],
                            'skills': processed_job['skills'][:5],  # First 5 skills
                            'apply_links': processed_job.get('apply_links', []),
                            'posted_date': processed_job.get('posted_date')
                        }
                    })
                    
                    # Immediately run matching for this job if we have resumes (only for new jobs)
                    await self.match_job_incrementally(processed_job)
            
            return processed_job
            
        except Exception as e:
            logger.error(f"Error processing job: {e}")
            self.emit('job_error', {
                'message': f"Error processing job: {str(e)}",
                'job_index': raw_job.get('job_index', 0)
            })
            return None
    
    async def match_job_incrementally(self, job: Dict):
        """
        Match a single job against all resumes immediately
        
        Args:
            job: Processed job data with job_id
        """
        try:
            # Get all resumes from database
            resumes = self.db.get_all_resumes()
            if not resumes:
                return
            
            job_id = job.get('job_id')
            if not job_id:
                return
            
            self.emit('matching_started', {
                'message': f"Matching {job['title']} against {len(resumes)} resumes",
                'job_id': job_id
            })
            
            # Run matching for this job
            matches = []
            for resume in resumes:
                try:
                    # Use the _match_single method from EnhancedMatcher
                    # which returns a DetailedMatch object with LLM assessment
                    logger.info(f"Running enhanced matching with LLM for {resume.get('name', 'Unknown')} -> {job.get('title')}")
                    
                    # Emit matching progress
                    self.emit('matching_progress', {
                        'message': f"Analyzing match: {resume.get('name', 'Unknown')}",
                        'job_id': job_id,
                        'resume_name': resume.get('name', 'Unknown')
                    })
                    
                    detailed_match = self.enhanced_matcher._match_single(
                        resume=resume,
                        job=job
                    )
                    
                    # Log and emit LLM assessment results
                    if detailed_match and detailed_match.llm_score:
                        logger.info(f"LLM assessment completed: score={detailed_match.llm_score:.2f}, reasoning={detailed_match.llm_reasoning[:100]}")
                        self.emit('llm_assessment_complete', {
                            'message': f"LLM analysis complete for {resume.get('name', 'Unknown')}",
                            'llm_score': detailed_match.llm_score,
                            'has_reasoning': bool(detailed_match.llm_reasoning)
                        })
                    
                    if detailed_match and detailed_match.overall_score >= 0.3:  # Min threshold
                        match_data = {
                            'job_id': job_id,
                            'resume_id': resume.get('resume_id'),
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
                            'job_title': job.get('title'),
                            'company': job.get('company'),
                            'resume_name': resume.get('name', 'Unknown'),
                            'matched_skills': detailed_match.matched_skills,
                            'skills_gap': detailed_match.missing_skills,  # Database expects skills_gap not missing_skills
                            'requirements_matched': detailed_match.matched_requirements if hasattr(detailed_match, 'matched_requirements') else [],
                            'requirements_gap': detailed_match.missing_requirements if hasattr(detailed_match, 'missing_requirements') else [],
                            'location_preference_met': detailed_match.location_preference_met,
                            'remote_preference_met': detailed_match.remote_preference_met,
                            'salary_match': detailed_match.salary_match,
                            'education_gaps': detailed_match.education_gaps,
                            'match_timestamp': datetime.now().isoformat()
                        }
                        matches.append(match_data)
                
                except Exception as e:
                    logger.error(f"Error matching job {job_id} with resume {resume.get('resume_id')}: {e}")
            
            # Save matches to database
            if matches:
                saved_count = self.db.save_matches(matches)
                top_matches = sorted(matches, key=lambda x: x['overall_score'], reverse=True)[:3]
                
                # Include LLM assessment info in top matches for UI display
                for match in top_matches:
                    if match.get('llm_reasoning'):
                        logger.info(f"Match includes LLM analysis: {match['resume_name']} -> {match['job_title']}")
                
                self.emit('matches_found', {
                    'message': f"Found {len(matches)} matches for {job['title']}",
                    'job_id': job_id,
                    'match_count': len(matches),
                    'top_matches': top_matches
                })
            
        except Exception as e:
            logger.error(f"Error in incremental matching: {e}")
    
    async def crawl_and_process_jobs(
        self,
        query: str,
        location: str = "remote",
        max_jobs: int = 10
    ):
        """
        Crawl and process jobs incrementally
        
        Args:
            query: Search query
            location: Job location
            max_jobs: Maximum jobs to crawl
        """
        try:
            # Start crawling
            self.emit('search_started', {
                'message': f"Searching for {query} in {location}",
                'query': query,
                'location': location,
                'max_jobs': max_jobs
            })
            
            crawler = StreamingCrawler(headless=True)
            
            # Callback to process each job as it's found
            async def on_job_found(raw_job):
                await self.process_job_incrementally(raw_job)
            
            # Crawl and process jobs incrementally
            job_count = 0
            async for raw_job in crawler.crawl_jobs_incrementally(
                query=query,
                location=location,
                max_jobs=max_jobs,
                on_job_found=on_job_found
            ):
                job_count += 1
                
            self.emit('search_completed', {
                'message': f"Completed: Found and processed {job_count} jobs",
                'total_jobs': job_count
            })
            
        except Exception as e:
            logger.error(f"Error in streaming pipeline: {e}")
            self.emit('search_error', {
                'message': f"Search error: {str(e)}"
            })


def emit_to_flask(event_type: str, data: Dict):
    """Emit events that Flask can process"""
    import sys
    
    # Format messages based on event type
    if event_type == 'search_started':
        print(f"Starting job search: {data['message']}")
    elif event_type == 'job_crawled':
        print(f"Job {data.get('job_index', 0) + 1}: Crawling...")
    elif event_type == 'job_processing':
        print(f"Job {data.get('job_index', 0) + 1}: Processing with LLM...")
    elif event_type == 'job_saved':
        job = data.get('job', {})
        print(f"Saved: {job.get('title')} at {job.get('company')}")
        # Show apply links if found
        apply_links = job.get('apply_links', [])
        if apply_links:
            print(f"  Found {len(apply_links)} apply link(s)")
        if job.get('posted_date'):
            print(f"  Posted: {job.get('posted_date')}")
        # Also emit as JSON for structured parsing
        print(f"Job saved!")
    elif event_type == 'job_duplicate':
        job = data.get('job', {})
        print(f"⏭️ Skipped duplicate: {job.get('title')} at {job.get('company')}")
    elif event_type == 'matching_progress':
        print(f"  {data.get('message', '')}")
    elif event_type == 'llm_assessment_complete':
        if data.get('has_reasoning'):
            print(f"  {data.get('message', '')} (LLM Score: {data.get('llm_score', 0):.0%})")
    elif event_type == 'matches_found':
        print(f"Found {data.get('match_count', 0)} matches for job")
        # Emit match data for immediate display
        for match in data.get('top_matches', [])[:3]:
            llm_info = f" [LLM: {match.get('llm_score', 0):.0%}]" if match.get('llm_score') else ""
            print(f"  - Match: {match.get('resume_name')} (Score: {match.get('overall_score', 0):.0%}){llm_info}")
        # Emit consistent save message
        if data.get('match_count', 0) > 0:
            print(f"Saved {data.get('match_count', 0)} matches")
    elif event_type == 'search_completed':
        print(f"{data['message']}")
    elif event_type == 'search_error':
        print(f"Error: {data['message']}")
    else:
        print(f"{data.get('message', '')}")
    
    # Flush output for real-time updates
    sys.stdout.flush()


async def main():
    """Main entry point for streaming pipeline"""
    import argparse
    import sys
    
    # Configure logging to show INFO level messages
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    parser = argparse.ArgumentParser(description='Streaming job search and matching pipeline')
    parser.add_argument('--query', required=True, help='Job search query')
    parser.add_argument('--location', default='remote', help='Job location')
    parser.add_argument('--max-jobs', type=int, default=10, help='Maximum jobs to process')
    parser.add_argument('--standalone', action='store_true', help='Run in standalone mode (start own LLM server)')
    
    args = parser.parse_args()
    
    # Initialize pipeline with Flask-compatible emit
    # Only auto-start LLM server if running in standalone mode
    pipeline = StreamingPipeline(emit_callback=emit_to_flask, auto_start_llm=args.standalone)
    
    try:
        # Run the streaming pipeline
        await pipeline.crawl_and_process_jobs(
            query=args.query,
            location=args.location,
            max_jobs=args.max_jobs
        )
    except KeyboardInterrupt:
        print("\nSearch interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

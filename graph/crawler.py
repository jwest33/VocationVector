"""
Simplified Google Jobs crawler - clicks cards, expands descriptions, returns text
"""
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote_plus

from playwright.async_api import async_playwright


class BulkJobsCrawler:
    """Crawler that expands all jobs then returns all content at once"""
    
    def __init__(self, headless: bool = True, data_dir: str = "data/jobs"):
        self.headless = headless
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    async def get_all_jobs_expanded(
        self, 
        query: str, 
        location: str = "",
        max_jobs: int = 10,
        max_retries: int = 2
    ) -> Dict:
        """
        Click all job cards, expand all descriptions, return all content
        
        Args:
            query: Job search query
            location: Location for the search
            max_jobs: Maximum number of jobs to expand
            max_retries: Maximum number of retries if location prompt blocks
            
        Returns:
            Dictionary with all expanded content
        """
        for retry_num in range(max_retries + 1):
            try:
                result = await self._crawl_with_timeout(query, location, max_jobs)
                if result and 'error' not in result:
                    return result
                elif retry_num < max_retries:
                    print(f"Retry {retry_num + 1}/{max_retries} due to: {result.get('error', 'unknown error')}")
                    await asyncio.sleep(2)
            except asyncio.TimeoutError:
                if retry_num < max_retries:
                    print(f"Timeout occurred, retry {retry_num + 1}/{max_retries}")
                    await asyncio.sleep(2)
                else:
                    return {'error': 'Crawler timed out after retries'}
            except Exception as e:
                if retry_num < max_retries:
                    print(f"Error occurred: {e}, retry {retry_num + 1}/{max_retries}")
                    await asyncio.sleep(2)
                else:
                    return {'error': f'Crawler failed: {str(e)}'}
        
        return {'error': 'Failed to crawl jobs after all retries'}
    
    async def _crawl_with_timeout(self, query: str, location: str, max_jobs: int, timeout: int = 120) -> Dict:
        """Internal method with timeout wrapper"""
        try:
            return await asyncio.wait_for(
                self._do_crawl(query, location, max_jobs),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            raise
    
    async def _do_crawl(self, query: str, location: str, max_jobs: int) -> Dict:
        """Actual crawling logic with improved error handling"""
        browser = None
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=self.headless,
                    args=[
                        '--disable-blink-features=AutomationControlled',
                        '--disable-dev-shm-usage',
                        '--no-sandbox',
                        '--disable-setuid-sandbox',
                        '--window-size=1920,1080',
                        '--disable-gpu',  # Help with stability
                        '--disable-web-security',
                        '--disable-features=IsolateOrigins,site-per-process'
                    ]
                )
                
                context = await browser.new_context(
                    viewport={'width': 1920, 'height': 1080},
                    user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    locale='en-US',
                    permissions=[],  # Deny all permissions including geolocation
                    geolocation=None  # No geolocation
                )
                
                page = await context.new_page()
                
                # Set up handler to dismiss location prompts
                async def handle_dialog(dialog):
                    print(f"Dismissing dialog: {dialog.message}")
                    await dialog.dismiss()
                
                page.on("dialog", handle_dialog)
                
                # Navigate to Google Jobs with location in query to avoid prompts
                # Include location explicitly in the URL to prevent Google from asking
                # Always include 'jobs' in the search to ensure we get job results
                if 'jobs' not in query.lower():
                    query = f"{query} jobs"
                search_term = f"{query} {location}".strip()
                encoded_query = quote_plus(search_term)
                # Add location parameter to URL to preempt location prompt
                location_param = f"&near={quote_plus(location)}" if location else ""
                url = f"https://www.google.com/search?q={encoded_query}&hl=en{location_param}"
                
                print(f"Searching for: {search_term}")
                await page.goto(url, wait_until='domcontentloaded')  # Faster than networkidle
                await asyncio.sleep(0.3)  # Further optimized
                
                # Check for and handle location prompts/modals
                try:
                    # Look for common location prompt elements
                    location_prompts = [
                        "text='Update location'",
                        "text='Set location'",
                        "text='Use precise location'",
                        "text='Location settings'",
                        "button:has-text('Not now')",
                        "button:has-text('No thanks')",
                        "button:has-text('Skip')",
                        "[aria-label*='location']",
                        "[aria-label*='Location']"
                    ]
                    
                    for prompt_selector in location_prompts:
                        try:
                            element = page.locator(prompt_selector).first
                            if await element.is_visible(timeout=100):  # Reduced from 500ms
                                # Try to find and click dismiss/skip button
                                dismiss_buttons = [
                                    "button:has-text('Not now')",
                                    "button:has-text('No thanks')",
                                    "button:has-text('Skip')",
                                    "button:has-text('Cancel')",
                                    "[aria-label*='Close']",
                                    "[aria-label*='Dismiss']"
                                ]
                                
                                for btn_selector in dismiss_buttons:
                                    try:
                                        btn = page.locator(btn_selector).first
                                        if await btn.is_visible(timeout=100):  # Reduced from 500ms
                                            await btn.click()
                                            print(f"Dismissed location prompt with: {btn_selector}")
                                            await asyncio.sleep(0.2)  # Reduced from 1 second
                                            break
                                    except:
                                        pass
                                break
                        except:
                            pass
                    
                    # Also try to close any overlay/modal by pressing Escape
                    await page.keyboard.press('Escape')
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    # Location prompt handling failed, but continue anyway
                    print(f"Note: Could not handle location prompt: {e}")
                
                # Click Jobs tab
                try:
                    jobs_tab = page.locator("a:has-text('Jobs')").first
                    if await jobs_tab.is_visible(timeout=1000):  # Reduced from 2000ms
                        print("Opening Google jobs tab...")
                        await jobs_tab.click()
                        await asyncio.sleep(0.5)  # Further optimized
                except:
                    jobs_url = url.replace("/search?", "/search?ibp=htl;jobs&")
                    await page.goto(jobs_url, wait_until='domcontentloaded')  # Faster
                    await asyncio.sleep(0.3)  # Further optimized
                
                # Find all job cards - use the correct selector
                working_selector = 'div.EimVGf'
                cards = await page.query_selector_all(working_selector)
                
                if not cards:
                    # Fallback to other selectors
                    for selector in ['li.iFjolb', 'div.PwjeAc', 'div[role="listitem"]']:
                        cards = await page.query_selector_all(selector)
                        if cards:
                            working_selector = selector
                            break
                
                if not cards:
                    print("No job cards found")
                    return {'error': 'No jobs found'}
                
                print(f"Found {len(cards)} job cards using selector: {working_selector}")
                
                # Process each job card
                jobs_to_process = min(len(cards), max_jobs)
                all_jobs = []
                
                print(f"Processing {jobs_to_process} jobs...")
                
                for i in range(jobs_to_process):
                    try:
                        print(f"\n  Processing job {i+1}/{jobs_to_process}")
                        
                        # Re-query cards to avoid stale references
                        current_cards = await page.query_selector_all(working_selector)
                        if i >= len(current_cards):
                            print(f"    Card {i} no longer available")
                            continue
                        
                        # Get preview and metadata from the card
                        card_text = await current_cards[i].text_content()
                        card_preview = (card_text[:60] + "...") if card_text and len(card_text) > 60 else card_text
                        
                        # Extract all metadata from Yf9oye elements (Google Jobs metadata tags)
                        job_metadata = await current_cards[i].evaluate("""
                            (element) => {
                                const metadata = {
                                    posted_date: null,
                                    location: null,
                                    employment_type: null,
                                    via: null,
                                    other: []
                                };
                                
                                // Find all Yf9oye class elements (Google's metadata containers)
                                const metadataElements = element.querySelectorAll('.Yf9oye');
                                
                                metadataElements.forEach(el => {
                                    const text = (el.textContent || '').trim();
                                    if (!text) return;
                                    
                                    // Parse different types of metadata
                                    if (text.match(/\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago|just posted|today|yesterday/i)) {
                                        // This is a date
                                        metadata.posted_date = text;
                                    } else if (text.match(/via\\s+/i)) {
                                        // This is the source (via Indeed, via LinkedIn, etc)
                                        metadata.via = text;
                                    } else if (text.match(/full[\\s-]?time|part[\\s-]?time|contract|temporary|intern|freelance/i)) {
                                        // This is employment type
                                        metadata.employment_type = text;
                                    } else if (text.match(/remote|hybrid|on[\\s-]?site/i) || 
                                              (text.includes(',') && !text.includes('via')) || 
                                              text.match(/[A-Z][a-z]+,\\s*[A-Z]{2}/)) {
                                        // This is likely a location
                                        metadata.location = text;
                                    } else {
                                        // Other metadata we might want to capture
                                        metadata.other.push(text);
                                    }
                                });
                                
                                // If no posted date found in Yf9oye, try other methods
                                if (!metadata.posted_date) {
                                    // Look for Date posted label
                                    const allText = element.innerText || '';
                                    const dateMatch = allText.match(/Date posted[:\\s]*(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago|just posted|today|yesterday)/i);
                                    if (dateMatch && dateMatch[1]) {
                                        metadata.posted_date = dateMatch[1].trim();
                                    }
                                }
                                
                                return metadata;
                            }
                        """)
                        
                        print(f"    Clicking card: {card_preview}")
                        if job_metadata.get('posted_date'):
                            print(f"    Posted: {job_metadata['posted_date']}")
                        if job_metadata.get('employment_type'):
                            print(f"    Type: {job_metadata['employment_type']}")
                        if job_metadata.get('location'):
                            print(f"    Location: {job_metadata['location']}")
                        
                        # Click the card
                        await current_cards[i].click()
                        await asyncio.sleep(0.5)  # Wait for details to load (OPTIMIZED)
                        
                        # Wait for job details to fully load
                        await asyncio.sleep(0.5)
                        
                        # Try to click "Show full description" using multiple methods
                        try:
                            # Method 1: Look for the button by text
                            show_full_btn = page.locator("text='Show full description'").first
                            if await show_full_btn.is_visible(timeout=1000):
                                await show_full_btn.click()
                                print(f"    Clicked 'Show full description'")
                                await asyncio.sleep(0.15)  # Wait for expansion (OPTIMIZED) - reduced from 1s
                        except:
                            pass
                        
                        # Try to click "More job highlights" 
                        try:
                            more_highlights_btn = page.locator("text='More job highlights'").first
                            if await more_highlights_btn.is_visible(timeout=1000):
                                await more_highlights_btn.click()
                                print(f"    Clicked 'More job highlights'")
                                await asyncio.sleep(0.15)  # Wait for expansion (OPTIMIZED) - reduced from 1s
                        except:
                            pass
                        
                        # Also try JavaScript click as fallback
                        expansions = await page.evaluate("""
                            () => {
                                let clicked = 0;
                                
                                // Find and click all expansion elements
                                const spans = document.querySelectorAll('span');
                                for (const span of spans) {
                                    const text = span.textContent || '';
                                    if (text.trim() === 'Show full description' || 
                                        text.trim() === 'More job highlights') {
                                        // Click the parent element (usually the clickable area)
                                        const clickTarget = span.parentElement || span;
                                        try {
                                            clickTarget.click();
                                            clicked++;
                                        } catch (e) {}
                                    }
                                }
                                
                                return clicked;
                            }
                        """)
                        
                        if expansions > 0:
                            print(f"    Clicked {expansions} additional expansion elements")
                            await asyncio.sleep(0.3)  # Wait for all expansions - reduced from 1s
                        
                        # Extract text, apply links, and posted date for this specific job
                        job_data = await page.evaluate("""
                            () => {
                                // Look for the active job detail panel
                                // Google shows ALL jobs in DOM, we need the visible one
                                
                                // Find the largest visible div with job content
                                const allDivs = document.querySelectorAll('div');
                                let bestMatch = '';
                                let largestSize = 0;
                                let applyLinks = [];
                                let bestDiv = null;
                                let postedDate = null;
                                
                                for (const div of allDivs) {
                                    const rect = div.getBoundingClientRect();
                                    // Must be visible
                                    if (rect.width === 0 || rect.height === 0) continue;
                                    
                                    const text = div.innerText || '';
                                    
                                    // Must have job content
                                    if (text.length < 500) continue;
                                    if (!text.includes('Apply')) continue;
                                    
                                    // Look for job-specific content
                                    const hasJobContent = 
                                        (text.includes('Full-time') || text.includes('Part-time') || 
                                         text.includes('Contract') || text.includes('Remote')) &&
                                        text.includes('Apply');
                                    
                                    if (hasJobContent) {
                                        // Check if it's in the detail area (right side of screen)
                                        const isRightSide = rect.left > window.innerWidth / 3;
                                        
                                        // Prefer larger text blocks that are on the right side
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
                                    const datePostedMatch = allText.match(/Date posted[:\\s]*(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago|just posted|today|yesterday)/i);
                                    if (datePostedMatch && datePostedMatch[1]) {
                                        postedDate = datePostedMatch[1].trim();
                                    } else {
                                        // Fallback to other date patterns
                                        const datePatterns = [
                                            /Posted[:\\s]*(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago)/gi,
                                            /(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago)(?=\\s*·|\\s*\\||\\s*-|$)/gi,
                                            /(just posted|today|yesterday)/gi,
                                            /via\\s+[^\\s]+\\s+(\\d+\\+?\\s*(day|week|month)s?\\s*ago)/gi
                                        ];
                                        
                                        for (const pattern of datePatterns) {
                                            const match = allText.match(pattern);
                                            if (match && match[0]) {
                                                // Clean up the match to get just the date part
                                                const cleanMatch = match[0].replace(/^(Posted|via\\s+\\S+)\\s*/i, '').trim();
                                                postedDate = cleanMatch;
                                                break;
                                            }
                                        }
                                    }
                                }
                                
                                if (bestMatch) {
                                    // Clean up the text
                                    return {
                                        text: bestMatch
                                            .replace(/\\n+/g, ' ')
                                            .replace(/\\s+/g, ' ')
                                            .trim(),
                                        apply_links: applyLinks,
                                        posted_date: postedDate
                                    };
                                }
                                
                                // Fallback: Just get the first large visible text block with job content
                                for (const div of allDivs) {
                                    const rect = div.getBoundingClientRect();
                                    if (rect.width > 0 && rect.height > 0) {
                                        const text = div.innerText || '';
                                        if (text.length > 1000 && text.includes('Apply') && 
                                            (text.includes('Full-time') || text.includes('Part-time') || 
                                             text.includes('Remote'))) {
                                            
                                            // Try to find apply links in this div too
                                            const links = div.querySelectorAll('a');
                                            const fallbackLinks = [];
                                            for (const link of links) {
                                                const href = link.href;
                                                const linkText = (link.textContent || '').toLowerCase();
                                                if (href && (linkText.includes('apply') || href.includes('apply'))) {
                                                    fallbackLinks.push({
                                                        url: href,
                                                        text: link.textContent.trim()
                                                    });
                                                }
                                            }
                                            
                                            // Try to extract posted date from fallback div
                                            let fallbackDate = null;
                                            
                                            // First look for "Date posted" label
                                            const datePostedMatch = text.match(/Date posted[:\\s]*(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago|just posted|today|yesterday)/i);
                                            if (datePostedMatch && datePostedMatch[1]) {
                                                fallbackDate = datePostedMatch[1].trim();
                                            } else {
                                                // Fallback patterns
                                                const datePatterns = [
                                                    /Posted[:\\s]*(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago)/gi,
                                                    /(\\d+\\+?\\s*(day|week|month|hour|minute)s?\\s*ago)(?=\\s*·|\\s*\\||\\s*-|$)/gi,
                                                    /(just posted|today|yesterday)/gi
                                                ];
                                                for (const pattern of datePatterns) {
                                                    const match = text.match(pattern);
                                                    if (match && match[0]) {
                                                        fallbackDate = match[0].replace(/^Posted\\s*/i, '').trim();
                                                        break;
                                                    }
                                                }
                                            }
                                            
                                            return {
                                                text: text
                                                    .replace(/\\n+/g, ' ')
                                                    .replace(/\\s+/g, ' ')
                                                    .trim(),
                                                apply_links: fallbackLinks,
                                                posted_date: fallbackDate
                                            };
                                        }
                                    }
                                }
                                
                                return { text: '', apply_links: [], posted_date: null };
                            }
                        """)
                        
                        if job_data and job_data.get('text'):
                            job_text = job_data['text']
                            apply_links = job_data.get('apply_links', [])
                            
                            # Use posted date from details panel if not found in card metadata
                            final_posted_date = job_metadata.get('posted_date') or job_data.get('posted_date')
                            
                            all_jobs.append({
                                'job_index': i,
                                'text': job_text,
                                'text_length': len(job_text),
                                'preview': card_preview,
                                'posted_date': final_posted_date,
                                'location': job_metadata.get('location'),
                                'employment_type': job_metadata.get('employment_type'),
                                'via': job_metadata.get('via'),
                                'metadata_other': job_metadata.get('other', []),
                                'apply_links': apply_links
                            })
                            print(f"    Captured {len(job_text)} characters")
                            if apply_links:
                                print(f"    Found {len(apply_links)} apply link(s)")
                            if final_posted_date:
                                print(f"    Posted: {final_posted_date}")
                        else:
                            print(f"    WARNING: No text captured for job {i+1}")
                        
                    except Exception as e:
                        print(f"    ERROR with job {i+1}: {e}")
                        continue
                
                # Prepare result
                result = {
                    'query': query,
                    'location': location,
                    'timestamp': datetime.now().isoformat(),
                    'total_jobs_found': len(cards),
                    'jobs_processed': jobs_to_process,
                    'jobs_captured': len(all_jobs),
                    'jobs': all_jobs
                }
                
                # Save the result
                self._save_data(result)
                
                return result
                
        except Exception as e:
            print(f"Crawler error: {e}")
            return {'error': f'Crawler failed: {str(e)}'}
        finally:
            if browser:
                try:
                    await browser.close()
                except:
                    pass  # Browser might already be closed
    
    def _save_data(self, data: Dict):
        """Save the extracted data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bulk_jobs_{timestamp}.json"
        filepath = self.data_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSaved to {filepath}")
        
        # Also save to latest
        latest_path = self.data_dir / "bulk_latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


async def main():
    """Test the bulk crawler"""
    crawler = BulkJobsCrawler(headless=False)
    
    result = await crawler.get_all_jobs_expanded(
        query="python developer",
        location="remote",
        max_jobs=3
    )
    
    print(f"\nResults:")
    print(f"  Jobs found: {result.get('total_jobs_found', 0)}")
    print(f"  Jobs processed: {result.get('jobs_processed', 0)}")
    print(f"  Jobs captured: {result.get('jobs_captured', 0)}")
    
    # Show individual job info
    for job in result.get('jobs', []):
        print(f"\n  Job {job['job_index']+1}:")
        print(f"    Preview: {job['preview']}")
        print(f"    Text length: {job['text_length']} characters")


if __name__ == "__main__":
    asyncio.run(main())

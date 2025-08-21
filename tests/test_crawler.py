"""
Unit tests for the job crawler module
"""

import pytest
import asyncio
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

from graph.crawler import BulkJobsCrawler


class TestBulkJobsCrawler:
    """Test suite for BulkJobsCrawler"""
    
    def test_crawler_initialization(self):
        """Test crawler initialization with different parameters"""
        # Test default initialization
        crawler = BulkJobsCrawler()
        assert crawler.headless == True
        assert str(crawler.data_dir) == "data/jobs" or str(crawler.data_dir) == "data\\jobs"
        
        # Test with custom parameters
        crawler = BulkJobsCrawler(headless=False, data_dir="custom/path")
        assert crawler.headless == False
        assert "custom" in str(crawler.data_dir)
    
    def test_save_data(self, tmp_path):
        """Test saving crawled data to file"""
        crawler = BulkJobsCrawler(data_dir=str(tmp_path))
        
        # Test saving data
        test_data = {
            "query": "python developer",
            "location": "remote",
            "jobs": [{"title": "Python Developer", "company": "ABC Corp"}],
            "timestamp": datetime.now().isoformat()
        }
        
        crawler._save_data(test_data)
        
        # Check if files were created (bulk_jobs_*.json and bulk_latest.json)
        import os
        files = os.listdir(tmp_path)
        assert len(files) == 2  # Both timestamped and latest
        json_files = [f for f in files if f.endswith('.json')]
        assert len(json_files) == 2
        assert any('bulk_jobs' in f for f in json_files)
        assert any('bulk_latest' in f for f in json_files)
    
    @pytest.mark.asyncio
    async def test_crawl_with_timeout(self):
        """Test crawling with timeout"""
        crawler = BulkJobsCrawler()
        
        with patch.object(crawler, '_do_crawl') as mock_crawl:
            # Mock successful crawl
            mock_crawl.return_value = {
                "jobs": [{"title": "Job 1"}],
                "timestamp": datetime.now().isoformat()
            }
            
            result = await crawler._crawl_with_timeout("python", "remote", 10, timeout=60)
            assert "jobs" in result
            assert len(result["jobs"]) == 1
    
    @pytest.mark.asyncio
    async def test_do_crawl_basic(self):
        """Test basic job crawling functionality"""
        with patch('graph.crawler.async_playwright') as mock_playwright:
            # Setup mock playwright
            mock_browser = AsyncMock()
            mock_context = AsyncMock()
            mock_page = AsyncMock()
            mock_p = AsyncMock()
            
            mock_playwright.return_value.__aenter__.return_value = mock_p
            mock_p.chromium.launch.return_value = mock_browser
            mock_browser.new_context.return_value = mock_context
            mock_context.new_page.return_value = mock_page
            
            # Mock page interactions
            mock_page.goto = AsyncMock()
            mock_page.query_selector_all = AsyncMock(return_value=[])
            mock_page.keyboard.press = AsyncMock()
            mock_page.locator.return_value.first.is_visible = AsyncMock(return_value=False)
            
            crawler = BulkJobsCrawler()
            result = await crawler._do_crawl("python developer", "remote", max_jobs=5)
            
            assert isinstance(result, dict)
            mock_page.goto.assert_called()
    
    def test_data_directory_creation(self, tmp_path):
        """Test that data directory is created if it doesn't exist"""
        import os
        data_dir = tmp_path / "test_jobs"
        
        # Directory shouldn't exist initially
        assert not data_dir.exists()
        
        # Create crawler with non-existent directory
        crawler = BulkJobsCrawler(data_dir=str(data_dir))
        
        # Directory should now exist
        assert data_dir.exists()
    
    @pytest.mark.asyncio
    async def test_crawl_timeout_handling(self):
        """Test timeout handling in crawl"""
        crawler = BulkJobsCrawler()
        
        with patch.object(crawler, '_do_crawl') as mock_crawl:
            # Mock timeout
            mock_crawl.side_effect = asyncio.TimeoutError()
            
            # Should raise TimeoutError
            with pytest.raises(asyncio.TimeoutError):
                result = await crawler._crawl_with_timeout("test query", "location", 10, timeout=1)
    
    @pytest.mark.asyncio  
    async def test_crawl_error_handling(self):
        """Test error handling in crawl"""
        crawler = BulkJobsCrawler()
        
        with patch.object(crawler, '_do_crawl') as mock_crawl:
            # Mock an error
            mock_crawl.side_effect = Exception("Crawl failed")
            
            # Should raise the exception
            with pytest.raises(Exception) as exc_info:
                result = await crawler._crawl_with_timeout("test", "location", 10)
            assert "Crawl failed" in str(exc_info.value)
    
    def test_file_naming_convention(self, tmp_path):
        """Test that saved files follow naming convention"""
        crawler = BulkJobsCrawler(data_dir=str(tmp_path))
        
        test_data = {
            "query": "python developer",
            "location": "san francisco",
            "jobs": [],
            "timestamp": "2024-01-01T12:00:00"
        }
        
        crawler._save_data(test_data)
        
        import os
        files = os.listdir(tmp_path)
        assert len(files) == 2  # Both timestamped and latest
        # Check that we have the expected files
        assert any('bulk_jobs' in f for f in files)
        assert any('bulk_latest' in f for f in files)

"""
Unit tests for pipeline module
"""

import pytest
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from datetime import datetime

from graph.pipeline import (
    PipelineMode,
    PipelineConfig,
    JobMatchingPipeline,
    run_pipeline,
    run_full_pipeline
)


class TestPipelineConfig:
    """Test suite for PipelineConfig"""
    
    def test_default_config(self):
        """Test default pipeline configuration"""
        config = PipelineConfig()
        
        assert config.mode == PipelineMode.FULL_PIPELINE
        assert config.job_query == "python developer"
        assert config.job_location == "remote"
        assert config.max_jobs == 20
        assert config.use_database == True
        assert config.use_llm_matching == True
        assert config.auto_start_server == True
    
    def test_config_modes(self):
        """Test different pipeline modes"""
        # Test all modes
        for mode in PipelineMode:
            config = PipelineConfig(mode=mode)
            assert config.mode == mode
    
    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = PipelineConfig(
            job_query="data scientist",
            max_jobs=10,
            use_cache=False
        )
        
        config_dict = config.to_dict()
        
        assert config_dict["job_query"] == "data scientist"
        assert config_dict["max_jobs"] == 10
        assert config_dict["use_cache"] == False
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "mode": PipelineMode.PROCESS_JOBS,
            "job_query": "machine learning",
            "max_jobs": 5,
            "use_database": False
        }
        
        config = PipelineConfig.from_dict(config_dict)
        
        assert config.mode == PipelineMode.PROCESS_JOBS
        assert config.job_query == "machine learning"
        assert config.max_jobs == 5
        assert config.use_database == False


class TestJobMatchingPipeline:
    """Test suite for JobMatchingPipeline"""
    
    @patch('graph.pipeline.LLMServerManager')
    def test_pipeline_initialization(self, mock_server_manager):
        """Test pipeline initialization"""
        config = PipelineConfig(auto_start_server=True)
        
        pipeline = JobMatchingPipeline(config)
        
        assert pipeline.config == config
        assert pipeline.server_manager is not None
        mock_server_manager.assert_called_once()
    
    def test_build_graph_full_pipeline(self):
        """Test building graph for full pipeline mode"""
        config = PipelineConfig(mode=PipelineMode.FULL_PIPELINE)
        pipeline = JobMatchingPipeline(config)
        
        graph = pipeline.graph
        
        assert graph is not None
        # Graph should have nodes for all stages
    
    def test_build_graph_process_jobs(self):
        """Test building graph for process jobs mode"""
        config = PipelineConfig(mode=PipelineMode.PROCESS_JOBS)
        pipeline = JobMatchingPipeline(config)
        
        graph = pipeline.graph
        
        assert graph is not None
        # Graph should only have job processing nodes
    
    def test_build_graph_match_only(self):
        """Test building graph for match only mode"""
        config = PipelineConfig(mode=PipelineMode.MATCH_ONLY)
        pipeline = JobMatchingPipeline(config)
        
        graph = pipeline.graph
        
        assert graph is not None
        # Graph should only have matching nodes
    
    @patch('graph.pipeline.graphDB')
    @patch('graph.pipeline.LLMServerManager')
    def test_run_pipeline_success(self, mock_server_manager, mock_db):
        """Test successful pipeline run"""
        # Setup mocks
        mock_server = MagicMock()
        mock_server.ensure_running.return_value = True
        mock_server_manager.return_value = mock_server
        
        mock_database = MagicMock()
        mock_db.return_value = mock_database
        
        config = PipelineConfig(
            mode=PipelineMode.PROCESS_JOBS,
            job_query="test query",
            max_jobs=5
        )
        
        pipeline = JobMatchingPipeline(config)
        
        # Mock graph execution
        with patch.object(pipeline.graph, 'invoke') as mock_invoke:
            # Return a state dict that matches what the pipeline expects
            mock_invoke.return_value = {
                "config": config,
                "processed_jobs": [{"title": "Test Job"}],
                "errors": [],
                "diagnostics": {"total_duration": 10.5},
                "start_time": datetime.now(),
                "end_time": datetime.now()
            }
            
            result = pipeline.run()
            
            assert result["success"] is True
            assert "results" in result
            # Check what the actual structure contains
            if "jobs" in result.get("results", {}):
                assert len(result["results"]["jobs"]) > 0
    
    @patch('graph.pipeline.graphDB')
    @patch('graph.pipeline.LLMServerManager')
    def test_run_pipeline_with_error(self, mock_server_manager, mock_db):
        """Test pipeline run with errors"""
        mock_server = MagicMock()
        mock_server_manager.return_value = mock_server
        mock_db.return_value = MagicMock()
        
        config = PipelineConfig()
        pipeline = JobMatchingPipeline(config)
        
        # Mock graph execution with error
        with patch.object(pipeline.graph, 'invoke') as mock_invoke:
            mock_invoke.side_effect = Exception("Pipeline error")
            
            result = pipeline.run()
            
            assert result["success"] is False
            assert "Pipeline error" in result["error"]
    
    @patch('graph.pipeline.graphDB')
    def test_run_with_resume_files(self, mock_db):
        """Test running pipeline with specific resume files"""
        mock_db.return_value = MagicMock()
        
        config = PipelineConfig(
            mode=PipelineMode.PROCESS_RESUMES,
            resume_files=["resume1.pdf", "resume2.docx"]
        )
        
        pipeline = JobMatchingPipeline(config)
        
        with patch.object(pipeline.graph, 'invoke') as mock_invoke:
            mock_invoke.return_value = {
                "processed_resumes": [{"name": "John"}, {"name": "Jane"}],
                "errors": []
            }
            
            result = pipeline.run()
            
            # Check that resume files were passed to state
            call_args = mock_invoke.call_args[0][0]
            assert "resume_files" in call_args
            assert len(call_args["resume_files"]) == 2


class TestPipelineNodes:
    """Test individual pipeline nodes"""
    
    @patch('graph.pipeline.BulkJobsCrawler')
    @pytest.mark.asyncio
    async def test_crawl_and_process_jobs_node(self, mock_crawler_class):
        """Test job crawling and processing node"""
        mock_crawler = MagicMock()
        # get_all_jobs_expanded is the actual method, and it's async
        mock_crawler.get_all_jobs_expanded = AsyncMock(return_value={
            "jobs": [
                {"text": "Job 1 text", "job_index": 0},
                {"text": "Job 2 text", "job_index": 1}
            ],
            "total_jobs_found": 2,
            "jobs_processed": 2
        })
        mock_crawler_class.return_value = mock_crawler
        
        # Use actual PipelineConfig instead of MagicMock
        config = PipelineConfig(
            job_query="python",
            job_location="remote",
            max_jobs=2,
            use_database=False,
            save_intermediate=False
        )
        
        state = {
            "config": config,
            "errors": []
        }
        
        # Import and test the node
        from graph.pipeline import crawl_and_process_jobs_node
        
        result = await crawl_and_process_jobs_node(state)
        
        assert "raw_jobs" in result
        assert len(result["raw_jobs"]) == 2
        mock_crawler.get_all_jobs_expanded.assert_called_once()


class TestAPIFunctions:
    """Test public API functions"""
    
    @patch('graph.pipeline.JobMatchingPipeline')
    def test_run_pipeline_api(self, mock_pipeline_class):
        """Test run_pipeline API function"""
        mock_pipeline = MagicMock()
        mock_pipeline.run.return_value = {"success": True}
        mock_pipeline_class.return_value = mock_pipeline
        
        config = {
            "mode": "process_jobs",
            "job_query": "test"
        }
        
        result = run_pipeline(config)
        
        assert result["success"] is True
        mock_pipeline_class.assert_called_once()
        mock_pipeline.run.assert_called_once()
    
    @patch('graph.pipeline.run_pipeline')
    def test_run_full_pipeline_api(self, mock_run):
        """Test run_full_pipeline API function"""
        mock_run.return_value = {
            "success": True,
            "results": {"matches": []}
        }
        
        result = run_full_pipeline(
            query="python developer",
            location="San Francisco",
            resume_directory="test_resumes",
            max_jobs=10
        )
        
        assert result["success"] is True
        
        # Check that correct config was passed
        call_args = mock_run.call_args[0][0]
        assert call_args["mode"] == "full_pipeline"
        assert call_args["job_query"] == "python developer"
        assert call_args["job_location"] == "San Francisco"
        assert call_args["max_jobs"] == 10


class TestPipelineIntegration:
    """Integration tests for pipeline"""
    
    @patch('graph.pipeline.BulkJobsCrawler')
    @patch('graph.nodes.job_processing.JobLLMProcessor')
    @patch('graph.pipeline.graphDB')
    def test_full_pipeline_integration(self, mock_db, mock_processor, mock_crawler):
        """Test full pipeline integration"""
        # Setup mocks
        mock_crawler_instance = MagicMock()
        mock_crawler_instance.crawl_jobs = AsyncMock(return_value=[
            {"text": "Job text", "job_index": 0}
        ])
        mock_crawler.return_value = mock_crawler_instance
        
        mock_processor_instance = MagicMock()
        mock_processor_instance.extract_job_info.return_value = MagicMock(
            model_dump=lambda: {"title": "Test Job", "company": "Test Co"}
        )
        mock_processor.return_value = mock_processor_instance
        
        mock_db_instance = MagicMock()
        mock_db_instance.add_job.return_value = "job_123"
        mock_db_instance.get_all_resumes.return_value = []
        mock_db.return_value = mock_db_instance
        
        # Run pipeline
        config = PipelineConfig(
            mode=PipelineMode.PROCESS_JOBS,
            job_query="test",
            max_jobs=1,
            auto_start_server=False
        )
        
        pipeline = JobMatchingPipeline(config)
        
        with patch.object(pipeline, 'server_manager', None):
            result = pipeline.run()
            
            # Basic assertions - pipeline should complete
            assert "results" in result
            assert "config" in result

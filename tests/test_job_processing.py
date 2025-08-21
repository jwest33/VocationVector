"""
Unit tests for job processing module
"""

import pytest
import json
import sys
from unittest.mock import Mock, MagicMock, patch

from graph.nodes.job_processing import (
    JobLLMProcessor,
    JobExtraction,
    JobProcessingConfig,
    job_processing_node
)


class TestJobProcessingConfig:
    """Test suite for JobProcessingConfig"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = JobProcessingConfig()
        assert config.DEFAULT_LLM_MODEL == "qwen3-4b-instruct-2507-f16"
        assert config.LLM_TEMPERATURE == 0.1
        assert config.LLM_MAX_TOKENS == 10000
        assert config.LLM_TIMEOUT == 240
        assert config.DEFAULT_BASE_URL == "http://localhost:8000/v1"
    
    def test_config_from_dict(self):
        """Test creating config from dictionary"""
        config_dict = {
            "LLM_TEMPERATURE": 0.5,
            "LLM_MAX_TOKENS": 5000,
            "LLM_TIMEOUT": 120
        }
        config = JobProcessingConfig.from_dict(config_dict)
        assert config.LLM_TEMPERATURE == 0.5
        assert config.LLM_MAX_TOKENS == 5000
        assert config.LLM_TIMEOUT == 120


class TestJobLLMProcessor:
    """Test suite for JobLLMProcessor"""
    
    def test_initialization(self, mock_llm_client):
        """Test processor initialization"""
        with patch('graph.nodes.job_processing.OpenAI', return_value=mock_llm_client):
            processor = JobLLMProcessor()
            assert processor.model is not None
            assert processor.base_url is not None
            assert processor.client is not None
    
    def test_parse_salary(self):
        """Test salary parsing functionality"""
        processor = JobLLMProcessor()
        
        # Test various salary formats
        assert processor._parse_salary("$100,000") == 100000
        assert processor._parse_salary("100000") == 100000
        assert processor._parse_salary("100k") == 100000
        assert processor._parse_salary("$50/hour") == 104000  # 50 * 2080
        assert processor._parse_salary("8000/month") == 96000  # 8000 * 12
        assert processor._parse_salary("invalid") is None
        assert processor._parse_salary(None) is None
    
    def test_parse_llm_response(self):
        """Test LLM response parsing"""
        processor = JobLLMProcessor()
        
        # Test valid JSON
        response = '{"title": "Python Developer", "company": "ABC Corp"}'
        parsed = processor._parse_llm_response(response)
        assert parsed["title"] == "Python Developer"
        assert parsed["company"] == "ABC Corp"
        
        # Test JSON with markdown
        response = '```json\n{"title": "Python Developer"}\n```'
        parsed = processor._parse_llm_response(response)
        assert parsed["title"] == "Python Developer"
        
        # Test invalid JSON - returns minimal structure
        response = "not json"
        parsed = processor._parse_llm_response(response)
        assert isinstance(parsed, dict)
        # Check for minimal structure fields
        assert "title" in parsed
        assert "company" in parsed
        assert "technical_skills" in parsed
        
        # Test truncated JSON
        response = '{"title": "Python Developer", "company": "ABC'
        parsed = processor._parse_llm_response(response)
        assert isinstance(parsed, dict)
    
    def test_call_llm_with_health_check(self, mock_llm_client):
        """Test LLM call with health check"""
        # Mock requests module
        sys.modules['requests'] = MagicMock()
        import requests
        mock_get = MagicMock()
        mock_get.return_value = Mock(status_code=200)
        requests.get = mock_get
        
        with patch('graph.nodes.job_processing.OpenAI', return_value=mock_llm_client):
            processor = JobLLMProcessor()
            
            # Mock LLM response
            mock_llm_client.chat.completions.create.return_value = Mock(
                choices=[Mock(message=Mock(content='{"test": "response"}'))]
            )
            
            result = processor._call_llm("system prompt", "user prompt")
            assert result == '{"test": "response"}'
            mock_get.assert_called()
    
    def test_analyze_job_domain(self, mock_llm_client):
        """Test job domain analysis"""
        with patch('graph.nodes.job_processing.OpenAI', return_value=mock_llm_client):
            processor = JobLLMProcessor()
            
            # Mock LLM to return domain analysis
            with patch.object(processor, '_call_llm') as mock_call:
                mock_call.return_value = json.dumps({
                    "domain": "payroll",
                    "level": "senior",
                    "key_skills_mentioned": ["ADP", "Workday"],
                    "has_technical_requirements": True,
                    "industry": "finance"
                })
                
                result = processor._analyze_job_domain("Payroll Manager job...")
                assert result["domain"] == "payroll"
                assert result["level"] == "senior"
                assert "ADP" in result["key_skills_mentioned"]
    
    def test_extract_basic_info(self, mock_llm_client, sample_job_text):
        """Test basic info extraction"""
        with patch('graph.nodes.job_processing.OpenAI', return_value=mock_llm_client):
            processor = JobLLMProcessor()
            
            with patch.object(processor, '_call_llm') as mock_call:
                mock_call.return_value = json.dumps({
                    "title": "Senior Payroll Manager",
                    "company": "ABC Company",
                    "location": "San Francisco, CA",
                    "salary_min": "85000",
                    "salary_max": "110000"
                })
                
                result = processor._extract_basic_info(sample_job_text)
                assert result["title"] == "Senior Payroll Manager"
                assert result["company"] == "ABC Company"
                assert result["salary_min"] == 85000
                assert result["salary_max"] == 110000
    
    def test_extract_skills(self, mock_llm_client):
        """Test skills extraction"""
        with patch('graph.nodes.job_processing.OpenAI', return_value=mock_llm_client):
            processor = JobLLMProcessor()
            
            with patch.object(processor, '_call_llm') as mock_call:
                mock_call.return_value = json.dumps({
                    "technical_skills": ["Python", "SQL", "AWS"],
                    "soft_skills": ["Leadership", "Communication"],
                    "tools_technologies": ["Docker", "Kubernetes"],
                    "certifications": ["AWS Certified"]
                })
                
                result = processor._extract_skills("Job text...", "tech")
                assert "Python" in result["technical_skills"]
                assert "Leadership" in result["soft_skills"]
                assert "Docker" in result["tools_technologies"]
    
    def test_extract_job_info_full(self, mock_llm_client, sample_job_text):
        """Test full job information extraction"""
        with patch('graph.nodes.job_processing.OpenAI', return_value=mock_llm_client):
            processor = JobLLMProcessor()
            
            # Mock all extraction methods
            with patch.object(processor, '_extract_basic_info') as mock_basic:
                with patch.object(processor, '_analyze_job_domain') as mock_domain:
                    with patch.object(processor, '_extract_skills') as mock_skills:
                        with patch.object(processor, '_extract_requirements') as mock_reqs:
                            
                            mock_basic.return_value = {
                                "title": "Senior Payroll Manager",
                                "company": "ABC Company"
                            }
                            mock_domain.return_value = {"domain": "payroll"}
                            mock_skills.return_value = {
                                "technical_skills": ["ADP", "Workday"]
                            }
                            mock_reqs.return_value = {
                                "responsibilities": ["Process payroll"],
                                "experience_requirements": ["5+ years"]
                            }
                            
                            result = processor.extract_job_info(sample_job_text)
                            
                            assert isinstance(result, JobExtraction)
                            assert result.title == "Senior Payroll Manager"
                            assert result.company == "ABC Company"


class TestJobProcessingNode:
    """Test suite for job processing node"""
    
    def test_job_processing_node_success(self, sample_job_text, mock_database):
        """Test successful job processing through node"""
        from dataclasses import dataclass, field
        from enum import Enum
        
        # Create minimal dataclasses for testing
        @dataclass
        class MockSkill:
            skill_name: str = "Python"
            is_mandatory: bool = True
            
        @dataclass
        class MockExperience:
            description: str = "5+ years"
            years: int = 5
            
        @dataclass  
        class MockEducation:
            degree_level: str = "Bachelor"
            field_of_study: str = "Computer Science"
            is_mandatory: bool = True
            
        class MockWorkArrangement:
            value = "remote"
            
        @dataclass
        class MockLocation:
            work_arrangement: MockWorkArrangement = field(default_factory=MockWorkArrangement)
            visa_sponsorship_needed: bool = False
            location: str = None
            timezone_preference: str = None
            
        @dataclass
        class MockCompensation:
            minimum_salary: int = 50000
            maximum_salary: int = 100000
            currency: str = "USD"
            benefits_required: list = field(default_factory=list)
            benefits_nice_to_have: list = field(default_factory=list)
            
        @dataclass
        class MockCultureFit:
            values: list = field(default_factory=list)
            work_style: str = ""
            
        class MockEmploymentType(Enum):
            FULL_TIME = "full_time"
            
        @dataclass
        class MockTemplate:
            source_type: str = "job_posting"
            source_id: str = "test_id"
            title: str = "Test Job"
            summary: str = "Test summary"
            technical_skills: list = field(default_factory=list)
            soft_skills: list = field(default_factory=list)
            certifications: list = field(default_factory=list)
            tools_technologies: list = field(default_factory=list)
            experience_requirements: list = field(default_factory=list)
            total_years_experience: int = 5
            management_experience: bool = False
            education_requirements: list = field(default_factory=list)
            location: MockLocation = field(default_factory=MockLocation)
            compensation: MockCompensation = field(default_factory=MockCompensation)
            employment_type: MockEmploymentType = MockEmploymentType.FULL_TIME
            start_date: str = ""
            culture_fit: MockCultureFit = field(default_factory=MockCultureFit)
            key_responsibilities: list = field(default_factory=list)
            achievements: list = field(default_factory=list)
            metadata: dict = field(default_factory=lambda: {"company": "Test Company"})
            extraction_confidence: float = 0.9
            processed_at: str = "2024-01-01T12:00:00"
            
        state = {
            "config": MagicMock(
                use_database=True,
                save_intermediate=False,
                output_directory="test_output"
            ),
            "raw_jobs": [{"text": sample_job_text, "job_index": 0}],
            "errors": [],
            "diagnostics": {},
            "database": mock_database
        }
        
        with patch('graph.nodes.job_processing.JobProcessingPipeline') as MockPipeline:
            mock_pipeline = MagicMock()
            MockPipeline.return_value = mock_pipeline
            
            # Create mock template instance
            mock_template = MockTemplate()
            
            mock_pipeline.process_jobs_batch.return_value = (
                [mock_template],
                {"jobs_processed": 1, "jobs_failed": 0}
            )
            
            result = job_processing_node(state)
            
            assert "processed_jobs" in result
            assert len(result["processed_jobs"]) == 1
            assert result["processed_jobs"][0]["title"] == "Test Job"
    
    def test_job_processing_node_error_handling(self):
        """Test error handling in job processing node"""
        state = {
            "config": MagicMock(use_database=False),
            "raw_jobs": [{"text": "invalid", "job_index": 0}],
            "errors": [],
            "diagnostics": {}
        }
        
        with patch('graph.nodes.job_processing.JobProcessingPipeline') as MockPipeline:
            mock_pipeline = MagicMock()
            MockPipeline.return_value = mock_pipeline
            # Make the processing fail
            mock_pipeline.process_jobs_batch.side_effect = Exception("Processing error")
            
            result = job_processing_node(state)
            
            assert result["processed_jobs"] == []
            assert "error" in result["diagnostics"]["job_processing"]["status"]

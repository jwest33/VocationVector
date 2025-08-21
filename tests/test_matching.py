"""
Unit tests for job matching modules
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from graph.nodes.job_matching import JobMatchingPipeline
from graph.nodes.enhanced_matching import EnhancedMatcher, DetailedMatch


class TestJobMatchingPipeline:
    """Test suite for JobMatchingPipeline"""
    
    def test_initialization(self):
        """Test pipeline initialization"""
        with patch('graph.nodes.job_matching.JobEmbeddings'):
            pipeline = JobMatchingPipeline()
            assert pipeline is not None
    
    def test_extract_job_title(self):
        """Test job title extraction"""
        with patch('graph.nodes.job_matching.JobEmbeddings'):
            pipeline = JobMatchingPipeline()
            
            # Test title extraction
            job_text = "Job Title: Senior Python Developer\nCompany: Tech Corp"
            title = pipeline._extract_job_title(job_text)
            assert title is not None
            
            # Test with no title
            job_text = "We are looking for someone to join our team"
            title = pipeline._extract_job_title(job_text)
            # May or may not find a title
    
    def test_extract_company(self):
        """Test company extraction"""
        with patch('graph.nodes.job_matching.JobEmbeddings'):
            pipeline = JobMatchingPipeline()
            
            # Test company extraction
            job_text = "Company: Google\nLocation: Mountain View"
            company = pipeline._extract_company(job_text)
            assert company is not None
            
            # Test with embedded company name
            job_text = "Join Microsoft as a Software Engineer"
            company = pipeline._extract_company(job_text)
            # May extract Microsoft
    
    def test_create_resume_text(self):
        """Test creating resume text from template"""
        with patch('graph.nodes.job_matching.JobEmbeddings'):
            pipeline = JobMatchingPipeline()
            
            # Use the correct structure expected by the method
            resume_template = {
                "qualifications": {
                    "experience_summary": "Experienced developer"
                },
                "requirements_match": {
                    "technical_skills": ["Python", "JavaScript"]
                },
                "work_history": [
                    {"title": "Senior Developer", "company": "Tech Corp"}
                ]
            }
            
            resume_text = pipeline._create_resume_text(resume_template)
            
            assert isinstance(resume_text, str)
            assert "Experienced developer" in resume_text
            assert "Python" in resume_text
    
    def test_get_job_id(self):
        """Test job ID extraction"""
        with patch('graph.nodes.job_matching.JobEmbeddings'):
            pipeline = JobMatchingPipeline()
            
            # Test with job_id field
            job = {"job_id": "job_123", "title": "Developer"}
            job_id = pipeline._get_job_id(job)
            assert job_id == "job_123"
            
            # Test with id field
            job = {"id": "456", "title": "Developer"}
            job_id = pipeline._get_job_id(job)
            assert job_id == "456"
            
            # Test with job_index field
            job = {"job_index": "789", "title": "Developer"}
            job_id = pipeline._get_job_id(job)
            assert job_id == "789"
            
            # Test with generated ID (uses hash)
            job = {"title": "Developer"}
            job_id = pipeline._get_job_id(job)
            # Hash returns a number, possibly negative
            assert isinstance(job_id, str)


class TestEnhancedMatcher:
    """Test suite for EnhancedMatcher"""
    
    def test_initialization(self):
        """Test enhanced matcher initialization"""
        with patch('graph.nodes.enhanced_matching.graphDB'):
            with patch('graph.nodes.enhanced_matching.JobEmbeddings'):
                matcher = EnhancedMatcher()
                assert matcher is not None
                assert hasattr(matcher, 'weights')
                assert sum(matcher.weights.values()) == pytest.approx(1.0, 0.01)
    
    def test_match_skills(self):
        """Test enhanced skills matching"""
        with patch('graph.nodes.enhanced_matching.graphDB'):
            with patch('graph.nodes.enhanced_matching.JobEmbeddings'):
                matcher = EnhancedMatcher()
                
                job = {
                    "skills": ["Python", "SQL", "AWS"],
                    "vocation_template": {
                        "technical_skills": [
                            {"skill_name": "Python", "required": True},
                            {"skill_name": "SQL", "required": True}
                        ]
                    }
                }
                
                resume = {
                    "skills": "[\"Python\", \"SQL\", \"Docker\"]"
                }

                result = matcher._match_skills(resume, job)

                assert "score" in result
                assert 0 <= result["score"] <= 1.0
    
    def test_match_experience(self):
        """Test enhanced experience matching"""
        with patch('graph.nodes.enhanced_matching.graphDB'):
            with patch('graph.nodes.enhanced_matching.JobEmbeddings'):
                matcher = EnhancedMatcher()
                
                job = {
                    "vocation_template": {
                        "years_experience_required": 5,
                        "experience_requirements": [
                            "5+ years in software development"
                        ]
                    }
                }
                
                resume = {
                    "experience": "[{\"duration\": \"2018 - Present\", \"title\": \"Software Engineer\"}]",
                    "years_experience": 5
                }
                
                result = matcher._match_experience(resume, job)
                assert "score" in result
                assert 0 <= result["score"] <= 1.0
    
    def test_match_education(self):
        """Test education matching"""
        with patch('graph.nodes.enhanced_matching.graphDB'):
            with patch('graph.nodes.enhanced_matching.JobEmbeddings'):
                matcher = EnhancedMatcher()
                
                job = {
                    "vocation_template": {
                        "education_requirements": [
                            {
                                "degree_level": "Bachelor",
                                "field_of_study": "Computer Science"
                            }
                        ]
                    }
                }
                
                resume = {
                    "education": "[{\"degree\": \"Bachelor of Science\", \"field\": \"Computer Science\"}]"
                }
                
                result = matcher._match_education(resume, job)
                assert "score" in result
                assert 0 <= result["score"] <= 1.0
    
    def test_match_location(self):
        """Test location matching"""
        with patch('graph.nodes.enhanced_matching.graphDB'):
            with patch('graph.nodes.enhanced_matching.JobEmbeddings'):
                matcher = EnhancedMatcher()
                
                job = {
                    "location": "San Francisco, CA",
                    "remote_policy": "hybrid"
                }
                
                resume = {
                    "location": "San Francisco, CA",
                    "work_preferences": "{\"preferred_locations\": [\"San Francisco\"]}"
                }
                
                result = matcher._match_location(resume, job)
                assert "score" in result
                assert 0 <= result["score"] <= 1.0
    
    def test_match_single_detailed(self):
        """Test detailed matching with all factors"""
        with patch('graph.nodes.enhanced_matching.graphDB'):
            with patch('graph.nodes.enhanced_matching.JobEmbeddings'):
                matcher = EnhancedMatcher()
                
                job = {
                    "job_id": "job_123",
                    "title": "Python Developer",
                    "company": "Tech Corp",
                    "location": "San Francisco, CA",
                    "remote_policy": "remote",
                    "skills": ["Python", "Django"],
                    "vocation_template": {
                        "years_experience_required": 3,
                        "technical_skills": [
                            {"skill_name": "Python", "required": True}
                        ],
                        "education_requirements": []
                    }
                }
                
                resume = {
                    "resume_id": "resume_456",
                    "name": "John Doe",
                    "skills": "[\"Python\", \"Django\", \"Flask\"]",
                    "experience": "[{\"title\": \"Developer\", \"duration\": \"2019-Present\"}]",
                    "education": "[{\"degree\": \"Bachelor\", \"field\": \"CS\"}]",
                    "years_experience": 4,
                    "location": "New York, NY",
                    "work_preferences": "{\"remote\": true}",
                    "vocation_template": {
                        "desired_roles": ["Developer", "Engineer"]
                    }
                }
                
                # Mock LLM response
                with patch.object(matcher, '_get_llm_assessment') as mock_llm:
                    mock_llm.return_value = {
                        "score": 0.85,
                        "reasoning": "Good match",
                        "recommendations": []
                    }
                    
                    match = matcher._match_single(resume, job)
                    
                    assert isinstance(match, DetailedMatch)
                    assert 0 <= match.overall_score <= 1.0
                    assert match.job_id == "job_123"
                    assert match.resume_id == "resume_456"

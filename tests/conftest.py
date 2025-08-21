"""
Pytest configuration and shared fixtures for all tests
"""

import os
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import pytest

# Mock heavy imports to speed up tests
import sys
sys.modules['tensorflow'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_job_text():
    """Sample job posting text for testing"""
    return """
    Senior Payroll Manager
    ABC Company - San Francisco, CA (Remote)
    
    We are seeking an experienced Payroll Manager to oversee our payroll operations.
    
    Requirements:
    - 5+ years of payroll experience
    - Experience with ADP and Workday
    - Knowledge of multi-state payroll regulations
    - Bachelor's degree in Accounting or related field
    
    Responsibilities:
    - Process bi-weekly payroll for 500+ employees
    - Ensure compliance with federal and state regulations
    - Manage payroll team of 3 specialists
    - Implement payroll system improvements
    
    Salary: $85,000 - $110,000
    Benefits: Health, dental, 401k, PTO
    """


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing"""
    return """
    John Doe
    john.doe@email.com | (555) 123-4567
    
    EXPERIENCE
    
    Payroll Specialist
    XYZ Corp | 2018 - Present
    - Process bi-weekly payroll for 300 employees
    - Proficient in ADP and QuickBooks
    - Handle multi-state tax filings
    
    Payroll Clerk
    DEF Company | 2016 - 2018
    - Assisted with payroll processing
    - Data entry and verification
    
    EDUCATION
    Bachelor of Science in Accounting
    State University | 2016
    
    SKILLS
    - ADP, Workday, QuickBooks
    - Multi-state payroll
    - Tax compliance
    - Excel, Python
    """


@pytest.fixture
def sample_job_extraction():
    """Sample job extraction result"""
    return {
        "title": "Senior Payroll Manager",
        "company": "ABC Company",
        "location": "San Francisco, CA",
        "remote_policy": "remote",
        "salary_min": 85000,
        "salary_max": 110000,
        "years_experience_required": 5,
        "technical_skills": [
            {"skill_name": "ADP", "required": True},
            {"skill_name": "Workday", "required": True},
            {"skill_name": "Payroll Processing", "required": True}
        ],
        "education_requirements": [
            {"degree_level": "Bachelor", "field_of_study": "Accounting"}
        ],
        "responsibilities": [
            "Process bi-weekly payroll for 500+ employees",
            "Ensure compliance with federal and state regulations",
            "Manage payroll team of 3 specialists"
        ]
    }


@pytest.fixture
def sample_resume_extraction():
    """Sample resume extraction result"""
    return {
        "name": "John Doe",
        "email": "john.doe@email.com",
        "phone": "(555) 123-4567",
        "skills": ["ADP", "Workday", "QuickBooks", "Multi-state payroll", "Tax compliance", "Excel", "Python"],
        "experience": [
            {
                "title": "Payroll Specialist",
                "company": "XYZ Corp",
                "duration": "2018 - Present",
                "responsibilities": [
                    "Process bi-weekly payroll for 300 employees",
                    "Proficient in ADP and QuickBooks",
                    "Handle multi-state tax filings"
                ]
            }
        ],
        "education": [
            {
                "degree": "Bachelor of Science",
                "field": "Accounting",
                "institution": "State University",
                "year": "2016"
            }
        ]
    }


@pytest.fixture
def mock_llm_client():
    """Mock OpenAI client for testing"""
    mock = MagicMock()
    mock.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps({
            "title": "Test Job",
            "company": "Test Company",
            "skills": ["Python", "SQL"]
        })))]
    )
    return mock


@pytest.fixture
def mock_database():
    """Mock database for testing"""
    mock = MagicMock()
    mock.add_job.return_value = "job_123"
    mock.add_resume.return_value = "resume_456"
    mock.save_matches.return_value = 1
    mock.get_all_jobs.return_value = []
    mock.get_all_resumes.return_value = []
    return mock


@pytest.fixture
def test_config():
    """Test configuration"""
    return {
        "environment": "testing",
        "database": {
            "db_path": "test_data/lancedb",
            "reset_if_exists": True
        },
        "llm": {
            "model_path": "test_model.gguf",
            "base_url": "http://localhost:8000/v1",
            "auto_start_server": False
        },
        "crawler": {
            "headless": True,
            "default_max_jobs": 3
        },
        "pipeline": {
            "output_directory": "test_data/output",
            "use_llm_matching": False,
            "use_cache": False
        }
    }


@pytest.fixture
def mock_embeddings():
    """Mock embeddings module"""
    mock = MagicMock()
    mock.generate_embedding.return_value = [0.1] * 768  # JobBERT-v2 dimension
    mock.generate_embeddings.return_value = [[0.1] * 768, [0.2] * 768]
    return mock

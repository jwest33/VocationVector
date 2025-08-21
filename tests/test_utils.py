"""
Utility functions and helpers for testing
"""

import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import random
import string


def create_temp_file(content: str, suffix: str = ".txt") -> str:
    """Create a temporary file with given content"""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def create_test_resume(name: str = None) -> Dict[str, Any]:
    """Create a test resume with realistic data"""
    if not name:
        name = f"Test User {random.randint(1, 100)}"
    
    return {
        "name": name,
        "email": f"{name.lower().replace(' ', '.')}@example.com",
        "phone": f"(555) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
        "skills": [
            "Python", "JavaScript", "SQL", "AWS", "Docker",
            "Git", "Agile", "REST APIs", "PostgreSQL"
        ],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Tech Corp",
                "duration": "2020 - Present",
                "responsibilities": [
                    "Developed scalable microservices",
                    "Led team of 5 engineers",
                    "Implemented CI/CD pipelines"
                ]
            },
            {
                "title": "Software Engineer",
                "company": "StartupXYZ",
                "duration": "2018 - 2020",
                "responsibilities": [
                    "Built RESTful APIs",
                    "Optimized database queries",
                    "Worked on frontend React components"
                ]
            }
        ],
        "education": [
            {
                "degree": "Bachelor of Science",
                "field": "Computer Science",
                "institution": "State University",
                "year": "2018"
            }
        ],
        "certifications": ["AWS Certified Developer", "Scrum Master"],
        "summary": "Experienced software engineer with expertise in full-stack development"
    }


def create_test_job(title: str = None, company: str = None) -> Dict[str, Any]:
    """Create a test job with realistic data"""
    if not title:
        title = random.choice([
            "Senior Software Engineer",
            "Data Scientist",
            "DevOps Engineer",
            "Product Manager",
            "Full Stack Developer"
        ])
    
    if not company:
        company = random.choice([
            "Tech Corp", "StartupXYZ", "Big Company Inc",
            "Innovation Labs", "Digital Solutions"
        ])
    
    return {
        "job_id": f"job_{random.randint(1000, 9999)}",
        "title": title,
        "company": company,
        "location": random.choice([
            "San Francisco, CA", "New York, NY", "Austin, TX",
            "Seattle, WA", "Remote"
        ]),
        "description": f"We are looking for a {title} to join our team...",
        "skills": random.sample([
            "Python", "Java", "JavaScript", "SQL", "AWS",
            "Docker", "Kubernetes", "React", "Node.js", "PostgreSQL",
            "MongoDB", "Redis", "Git", "CI/CD", "Agile"
        ], k=random.randint(5, 10)),
        "requirements": [
            f"{random.randint(3, 7)}+ years of experience",
            "Bachelor's degree or equivalent",
            "Strong problem-solving skills"
        ],
        "responsibilities": [
            "Design and develop software solutions",
            "Collaborate with cross-functional teams",
            "Mentor junior developers"
        ],
        "salary_min": random.randint(80, 120) * 1000,
        "salary_max": random.randint(130, 200) * 1000,
        "employment_type": random.choice(["full-time", "contract", "part-time"]),
        "remote_policy": random.choice(["remote", "hybrid", "onsite"]),
        "years_experience_required": random.randint(3, 10),
        "posted_date": datetime.now().isoformat()
    }


def create_test_match(job: Dict, resume: Dict) -> Dict[str, Any]:
    """Create a test match between job and resume"""
    # Calculate realistic scores based on overlap
    job_skills = set(job.get("skills", []))
    resume_skills = set(resume.get("skills", []))
    
    skills_overlap = len(job_skills & resume_skills) / max(len(job_skills), 1)
    
    return {
        "job_id": job.get("job_id", "job_unknown"),
        "resume_id": resume.get("resume_id", f"resume_{random.randint(1000, 9999)}"),
        "job_title": job.get("title"),
        "company": job.get("company"),
        "resume_name": resume.get("name"),
        "overall_score": min(skills_overlap + random.uniform(0.1, 0.3), 1.0),
        "skills_score": skills_overlap,
        "experience_score": random.uniform(0.5, 1.0),
        "education_score": random.uniform(0.6, 1.0),
        "location_score": random.uniform(0.4, 1.0),
        "matched_at": datetime.now().isoformat()
    }


def generate_random_text(min_words: int = 50, max_words: int = 200) -> str:
    """Generate random text for testing"""
    words = [
        "develop", "implement", "design", "analyze", "optimize",
        "manage", "collaborate", "build", "create", "maintain",
        "software", "system", "application", "database", "infrastructure",
        "team", "project", "solution", "platform", "service",
        "Python", "Java", "JavaScript", "SQL", "AWS",
        "experience", "skills", "requirements", "responsibilities", "qualifications"
    ]
    
    num_words = random.randint(min_words, max_words)
    text = " ".join(random.choices(words, k=num_words))
    
    # Add some structure
    sentences = []
    current = []
    for i, word in enumerate(text.split()):
        current.append(word)
        if i % random.randint(5, 15) == 0:
            sentences.append(" ".join(current).capitalize() + ".")
            current = []
    if current:
        sentences.append(" ".join(current).capitalize() + ".")
    
    return " ".join(sentences)


def compare_embeddings(emb1: List[float], emb2: List[float], tolerance: float = 0.01) -> bool:
    """Compare two embeddings with tolerance"""
    if len(emb1) != len(emb2):
        return False
    
    for a, b in zip(emb1, emb2):
        if abs(a - b) > tolerance:
            return False
    
    return True


def create_test_database_config() -> Dict[str, Any]:
    """Create test database configuration"""
    temp_dir = tempfile.mkdtemp()
    
    return {
        "db_path": temp_dir,
        "reset_if_exists": True,
        "auto_recreate_on_schema_error": True,
        "embedding_batch_size": 32,
        "max_search_results": 50
    }


def cleanup_test_files(*paths: str):
    """Clean up test files and directories"""
    for path in paths:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path_obj.unlink(missing_ok=True)


def assert_valid_job(job: Dict[str, Any]):
    """Assert that a job object has required fields"""
    required_fields = ["title", "company"]
    for field in required_fields:
        assert field in job, f"Job missing required field: {field}"
    
    # Check types
    assert isinstance(job.get("title"), str)
    assert isinstance(job.get("company"), str)
    
    if "skills" in job:
        assert isinstance(job["skills"], list)
    
    if "salary_min" in job:
        assert isinstance(job["salary_min"], (int, float))


def assert_valid_resume(resume: Dict[str, Any]):
    """Assert that a resume object has required fields"""
    required_fields = ["name"]
    for field in required_fields:
        assert field in resume, f"Resume missing required field: {field}"
    
    # Check types
    assert isinstance(resume.get("name"), str)
    
    if "skills" in resume:
        assert isinstance(resume["skills"], list)
    
    if "experience" in resume:
        assert isinstance(resume["experience"], list)


def assert_valid_match(match: Dict[str, Any]):
    """Assert that a match object has required fields"""
    required_fields = ["job_id", "resume_id", "overall_score"]
    for field in required_fields:
        assert field in match, f"Match missing required field: {field}"
    
    # Check score ranges
    assert 0 <= match["overall_score"] <= 1.0
    
    if "skills_score" in match:
        assert 0 <= match["skills_score"] <= 1.0

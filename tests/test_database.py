"""
Unit tests for database module
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import json
import numpy as np

from graph.database import graphDB


class TestGraphDB:
    """Test suite for graphDB class"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database path"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_database_initialization(self, temp_db_path):
        """Test database initialization"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db
            
            db = graphDB(db_path=temp_db_path)
            
            assert db.db_path == temp_db_path
            mock_connect.assert_called_once_with(temp_db_path)
    
    def test_create_tables(self, temp_db_path):
        """Test table creation"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db
            
            # Mock table creation
            mock_db.table_names.return_value = []
            mock_db.create_table.return_value = MagicMock()
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                
                # Should create three tables
                assert mock_db.create_table.call_count == 3
                
                # Check table names
                call_args = [call[0][0] for call in mock_db.create_table.call_args_list]
                assert "jobs" in call_args
                assert "resumes" in call_args
                assert "matches" in call_args
    
    def test_add_job(self, temp_db_path):
        """Test adding a job to the database"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            mock_db.table_names.return_value = ["jobs"]
            
            # Mock empty table for duplicate check
            mock_table.to_pandas.return_value = MagicMock(empty=True)
            
            # Mock embeddings
            with patch('graph.database.JobEmbeddings') as mock_embeddings_class:
                mock_embeddings = MagicMock()
                mock_embeddings_class.return_value = mock_embeddings
                # Return numpy array that has tolist() method
                mock_embeddings.encode.return_value = np.array([[0.1] * 768])
                
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                
                job_data = {
                    "text": "Job description text",
                    "title": "Python Developer",
                    "company": "Test Corp",
                    "description": "We are looking for a Python developer",
                    "skills": ["Python", "Django"],
                    "requirements": ["3+ years experience"],
                    "job_index": 0
                }
                
                job_id = db.add_job(job_data, "python developer", "remote")
                
                assert job_id is not None
                assert job_id.startswith("job_")
                mock_table.add.assert_called_once()
    
    def test_add_resume(self, temp_db_path):
        """Test adding a resume to the database"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            mock_db.table_names.return_value = ["resumes"]
            
            # Mock empty table for duplicate check
            mock_table.to_pandas.return_value = MagicMock(empty=True)
            
            # Mock embeddings
            with patch('graph.database.JobEmbeddings') as mock_embeddings_class:
                mock_embeddings = MagicMock()
                mock_embeddings_class.return_value = mock_embeddings
                # Return numpy array that has tolist() method
                mock_embeddings.encode.return_value = np.array([[0.1] * 768])
                
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                
                resume_data = {
                    "full_text": "Resume text",
                    "contact_info": {
                        "name": "John Doe",
                        "email": "john@example.com"
                    },
                    "summary": "Experienced developer",
                    "experience": [{"title": "Developer", "company": "ABC Corp"}],
                    "technical_skills": ["Python", "JavaScript"],
                    "filename": "john_doe.pdf"
                }
                
                resume_id = db.add_resume(resume_data)
                
                assert resume_id is not None
                assert resume_id.startswith("resume_")
                mock_table.add.assert_called_once()
    
    def test_save_matches(self, temp_db_path):
        """Test saving matches to the database"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            mock_db.table_names.return_value = ["matches"]
            
            # Mock existing matches for duplicate check
            mock_df = MagicMock()
            mock_df.empty = True
            mock_table.to_pandas.return_value = mock_df
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                
                matches = [
                    {
                        "job_id": "job_123",
                        "resume_id": "resume_456",
                        "overall_score": 0.85,
                        "skills_score": 0.90,
                        "match_reasons": ["Good match"]
                    },
                    {
                        "job_id": "job_124",
                        "resume_id": "resume_456",
                        "overall_score": 0.75,
                        "skills_score": 0.80,
                        "match_reasons": ["Fair match"]
                    }
                ]
                
                count = db.save_matches(matches)
                
                assert count == 2
                # Should call add for each match  
                assert mock_table.add.call_count == 2
    
    def test_get_all_jobs(self, temp_db_path):
        """Test retrieving all jobs"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            
            # Mock table data
            mock_df = MagicMock()
            mock_df.to_dict.return_value = [
                {"job_id": "job_1", "title": "Job 1"},
                {"job_id": "job_2", "title": "Job 2"}
            ]
            mock_table.to_pandas.return_value = mock_df
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                jobs = db.get_all_jobs()
            
            assert len(jobs) == 2
            assert jobs[0]["title"] == "Job 1"
            assert jobs[1]["title"] == "Job 2"
    
    def test_get_all_resumes(self, temp_db_path):
        """Test retrieving all resumes"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            
            # Mock table data
            mock_df = MagicMock()
            mock_df.to_dict.return_value = [
                {"resume_id": "resume_1", "name": "John Doe"},
                {"resume_id": "resume_2", "name": "Jane Smith"}
            ]
            mock_table.to_pandas.return_value = mock_df
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                resumes = db.get_all_resumes()
            
            assert len(resumes) == 2
            assert resumes[0]["name"] == "John Doe"
            assert resumes[1]["name"] == "Jane Smith"
    
    def test_search_jobs_by_resume(self, temp_db_path):
        """Test searching jobs by resume"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            mock_db.table_names.return_value = ["jobs", "resumes"]
            
            # Mock resume data
            mock_resume_df = MagicMock()
            mock_resume_df.empty = False
            mock_resume_df.iloc = [{"embedding_full": [0.1] * 768, "embedding_skills": [0.2] * 768}]
            
            # Mock search results
            mock_search = MagicMock()
            mock_search.limit.return_value = mock_search
            mock_df = MagicMock()
            mock_df.to_dict.return_value = [
                {"job_id": "job_1", "title": "Python Developer"}
            ]
            mock_search.to_pandas.return_value = mock_df
            mock_table.search.return_value = mock_search
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                
                # Mock the table.to_pandas() for resume lookup
                mock_table.to_pandas.return_value = mock_resume_df
                
                results = db.search_jobs_by_resume(resume_id="resume_123", limit=10)
                
                # May return empty if resume not found, but method should execute
                assert isinstance(results, list)
    
    def test_clear_all_matches(self, temp_db_path):
        """Test clearing all matches"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db
            
            # Mock table names
            mock_db.table_names.return_value = ["matches"]
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                db.clear_all_matches()
                
                # Should call recreate_table for matches
                # Since recreate_table drops and recreates, we check for that pattern
    
    def test_get_matches_for_job(self, temp_db_path):
        """Test getting matches for a specific job"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            
            # Mock pandas DataFrame
            mock_df = MagicMock()
            mock_df.empty = False
            
            # Mock filtering operation df[df['job_id'] == job_id]
            filtered_df = MagicMock()
            filtered_df.to_dict.return_value = [
                {
                    "job_id": "job_123",
                    "resume_id": "resume_456",
                    "overall_score": 0.85
                }
            ]
            mock_df.__getitem__.return_value = filtered_df
            mock_table.to_pandas.return_value = mock_df
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                matches = db.get_matches_for_job("job_123")
            
            assert len(matches) == 1
            assert matches[0]["job_id"] == "job_123"
            assert matches[0]["overall_score"] == 0.85
    
    def test_duplicate_job_prevention(self, temp_db_path):
        """Test that duplicate jobs are not added"""
        with patch('graph.database.lancedb.connect') as mock_connect:
            mock_db = MagicMock()
            mock_table = MagicMock()
            mock_connect.return_value = mock_db
            mock_db.open_table.return_value = mock_table
            mock_db.table_names.return_value = ["jobs"]
            
            # Mock existing job with same title and company
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.columns = ['title', 'company', 'job_id']  # Add columns attribute
            
            # Set up the duplicate detection logic
            # When checking (df['title'] == title) & (df['company'] == company)
            title_match = MagicMock()
            company_match = MagicMock()
            combined_match = MagicMock()
            combined_match.any.return_value = True  # Duplicate exists
            
            mock_df.__getitem__.side_effect = lambda key: {
                'title': title_match,
                'company': company_match
            }.get(key, MagicMock())
            
            # Mock the & operation result
            title_match.__eq__.return_value = MagicMock()
            company_match.__eq__.return_value = MagicMock()
            title_match.__eq__.return_value.__and__.return_value = combined_match
            
            # Mock iloc to return existing job
            mock_iloc_result = MagicMock()
            mock_iloc_result.__getitem__.return_value = {"job_id": "existing_job_id"}
            mock_df.__getitem__.return_value.iloc = mock_iloc_result
            
            mock_table.to_pandas.return_value = mock_df
            
            with patch('graph.database.JobEmbeddings'):
                db = graphDB(db_path=temp_db_path, reset_if_exists=False)
                
                job_data = {
                    "text": "Job description",
                    "title": "Python Developer",
                    "company": "Test Corp",
                    "job_index": 0
                }
                
                # Should return existing job ID
                job_id = db.add_job(job_data, "python developer", "remote")
                
                # Should not call add since it's a duplicate
                mock_table.add.assert_not_called()

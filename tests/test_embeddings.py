"""
Unit tests for embeddings module
"""

import pytest
import numpy as np
import os
from unittest.mock import Mock, MagicMock, patch

from graph.embeddings import JobEmbeddings


class TestJobEmbeddings:
    """Test suite for JobEmbeddings"""
    
    def setup_method(self):
        """Reset singleton before each test"""
        JobEmbeddings._instance = None
        JobEmbeddings._model = None
    
    @patch('graph.embeddings.SentenceTransformer')
    @patch.dict('os.environ', {'EMBEDDING_MODEL': 'test-model'})
    def test_initialization(self, mock_transformer):
        """Test embeddings initialization"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        
        assert embeddings.llm_name == "test-model"
        mock_transformer.assert_called_once_with("test-model")
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_default_model(self, mock_transformer):
        """Test default model initialization"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        
        assert embeddings.llm_name == "TechWolf/JobBERT-v2"
        mock_transformer.assert_called_once_with("TechWolf/JobBERT-v2")
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_encode_job(self, mock_transformer):
        """Test encoding job text"""
        mock_model = MagicMock()
        # encode returns array of shape (1, 768) for single text
        mock_embedding = np.array([[0.1, 0.2, 0.3] * 256])  # 768 dimensions
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        result = embeddings.encode_job("Python developer with 5 years experience")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)
        mock_model.encode.assert_called_once()
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_encode_batch_jobs(self, mock_transformer):
        """Test encoding multiple job texts"""
        mock_model = MagicMock()
        mock_embeddings = np.array([
            [0.1] * 768,
            [0.2] * 768,
            [0.3] * 768
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        texts = [
            "Python developer",
            "Data scientist",
            "DevOps engineer"
        ]
        
        result = embeddings.encode_batch_jobs(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
        mock_model.encode.assert_called_once()
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_encode_resume(self, mock_transformer):
        """Test encoding resume text"""
        mock_model = MagicMock()
        # encode returns array of shape (1, 768) for single text
        mock_embedding = np.array([[0.5] * 768])
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        result = embeddings.encode_resume("Experienced software engineer with Python expertise")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)
        mock_model.encode.assert_called_once()
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_encode_batch_resumes(self, mock_transformer):
        """Test encoding multiple resume texts"""
        mock_model = MagicMock()
        mock_embeddings = np.array([
            [0.4] * 768,
            [0.5] * 768,
            [0.6] * 768
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        texts = [
            "Software engineer with 5 years experience",
            "Data scientist with ML expertise",
            "DevOps engineer with cloud skills"
        ]
        
        result = embeddings.encode_batch_resumes(texts)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 768)
        mock_model.encode.assert_called_once()
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_encode_with_options(self, mock_transformer):
        """Test encoding with additional options"""
        mock_model = MagicMock()
        mock_embedding = np.array([[0.7] * 768])
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        
        # Test with show_progress option
        result = embeddings.encode_job("Test job")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)
        
        # Check that default options were passed
        call_kwargs = mock_model.encode.call_args[1]
        assert "show_progress_bar" in call_kwargs
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_cosine_similarity(self, mock_transformer):
        """Test cosine similarity calculation"""
        from sklearn.metrics.pairwise import cosine_similarity
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        
        # Test with identical embeddings
        emb1 = np.array([[1.0, 0.0, 0.0] * 256])
        emb2 = np.array([[1.0, 0.0, 0.0] * 256])
        
        score = cosine_similarity(emb1, emb2)[0][0]
        assert score == pytest.approx(1.0, 0.01)
        
        # Test with orthogonal embeddings
        emb1 = np.array([[1.0, 0.0, 0.0] * 256])
        emb2 = np.array([[0.0, 1.0, 0.0] * 256])
        
        score = cosine_similarity(emb1, emb2)[0][0]
        assert score == pytest.approx(0.0, 0.01)
        
        # Test with opposite embeddings
        emb1 = np.array([[1.0] * 768])
        emb2 = np.array([[-1.0] * 768])
        
        score = cosine_similarity(emb1, emb2)[0][0]
        assert score == pytest.approx(-1.0, 0.01)
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_batch_encode(self, mock_transformer):
        """Test batch encoding consistency"""
        mock_model = MagicMock()
        mock_embeddings = np.array([
            [0.1] * 768,
            [0.2] * 768
        ])
        mock_model.encode.return_value = mock_embeddings
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        
        # Encode jobs and resumes
        job_results = embeddings.encode_batch_jobs(["Job 1", "Job 2"])
        resume_results = embeddings.encode_batch_resumes(["Resume 1", "Resume 2"])
        
        assert job_results.shape == (2, 768)
        assert resume_results.shape == (2, 768)
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_model_caching(self, mock_transformer):
        """Test that model is loaded only once"""
        mock_model = MagicMock()
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        
        # Since it's a singleton, creating another instance should return the same object
        embeddings2 = JobEmbeddings()
        
        # Should be the same instance
        assert embeddings is embeddings2
        
        # Should only be loaded once
        mock_transformer.assert_called_once()
    
    @patch('graph.embeddings.SentenceTransformer')
    def test_encode_empty_text(self, mock_transformer):
        """Test encoding empty text"""
        mock_model = MagicMock()
        mock_embedding = np.zeros((1, 768))
        mock_model.encode.return_value = mock_embedding
        mock_transformer.return_value = mock_model
        
        embeddings = JobEmbeddings()
        result = embeddings.encode_job("")
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 768)
        mock_model.encode.assert_called_once()

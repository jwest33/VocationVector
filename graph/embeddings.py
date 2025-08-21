"""
Centralized embedding configuration for the job matching system
Uses JobBERT-v2 throughout for consistent job-specific understanding
"""

import os
import logging
from typing import List, Optional, Union
import numpy as np

# Suppress TensorFlow warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Set TensorFlow logging to ERROR only
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 1=INFO, 2=WARNING, 3=ERROR

# Import after setting environment variables
from sentence_transformers import SentenceTransformer

# Additional suppression for protobuf issues
import sys
import io
import contextlib

logger = logging.getLogger(__name__)


class EmbeddingConfig:
    """Central configuration for embeddings"""
    
    # Use JobBERT-v2 everywhere for job-specific semantic understanding
    # This model is specifically trained on job postings and resumes
    DEFAULT_MODEL = "TechWolf/JobBERT-v2"
    
    # Alternative models (if needed for specific use cases)
    ALTERNATIVE_MODELS = {
        "general": "sentence-transformers/all-MiniLM-L6-v2",  # Faster, general purpose
        "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # Multiple languages
        "large": "sentence-transformers/all-mpnet-base-v2",  # Higher quality, slower
    }
    
    # Batch processing settings
    DEFAULT_BATCH_SIZE = 32
    MAX_SEQUENCE_LENGTH = 512  # JobBERT-v2 max sequence length
    
    @classmethod
    def get_llm_name(cls, use_case: str = "default") -> str:
        """Get the appropriate model name for a use case"""
        if use_case == "default" or use_case == "job":
            return cls.DEFAULT_MODEL
        return cls.ALTERNATIVE_MODELS.get(use_case, cls.DEFAULT_MODEL)


class JobEmbeddings:
    """
    Unified embedding handler for the entire application
    Ensures consistent embeddings across all components
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Singleton pattern to reuse the same model instance"""
        if cls._instance is None:
            cls._instance = super(JobEmbeddings, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, llm_name: Optional[str] = None):
        """Initialize with JobBERT-v2 or custom model"""
        if self._model is None:
            self.llm_name = llm_name or os.getenv("EMBEDDING_MODEL", EmbeddingConfig.DEFAULT_MODEL)
            logger.info(f"Initializing embedding model: {self.llm_name}")
            
            # Suppress stderr output during model loading to avoid TensorFlow/protobuf warnings
            with contextlib.redirect_stderr(io.StringIO()):
                try:
                    self._model = SentenceTransformer(self.llm_name)
                except Exception as e:
                    # If there's an actual error, log it
                    logger.error(f"Error loading model {self.llm_name}: {e}")
                    raise
            
            logger.info(f"Loaded {self.llm_name} successfully")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the model instance"""
        if self._model is None:
            self.__init__()
        return self._model
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = None,
        show_progress: bool = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar for large batches
            normalize: Normalize embeddings to unit vectors
        
        Returns:
            Embeddings as numpy array
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Auto-detect when to show progress
        if show_progress is None:
            show_progress = len(texts) > 50
        
        if batch_size is None:
            batch_size = EmbeddingConfig.DEFAULT_BATCH_SIZE
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Calculate cosine similarity between embeddings
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
        
        Returns:
            Similarity matrix
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Ensure 2D arrays
        if len(embeddings1.shape) == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if len(embeddings2.shape) == 1:
            embeddings2 = embeddings2.reshape(1, -1)
        
        return cosine_similarity(embeddings1, embeddings2)
    
    def encode_job(self, job_text: str) -> np.ndarray:
        """Encode a job posting"""
        # Could add job-specific preprocessing here
        return self.encode(job_text)
    
    def encode_resume(self, resume_text: str) -> np.ndarray:
        """Encode a resume"""
        # Could add resume-specific preprocessing here
        return self.encode(resume_text)
    
    def encode_batch_jobs(self, job_texts: List[str], **kwargs) -> np.ndarray:
        """Encode multiple job postings"""
        return self.encode(job_texts, **kwargs)
    
    def encode_batch_resumes(self, resume_texts: List[str], **kwargs) -> np.ndarray:
        """Encode multiple resumes"""
        return self.encode(resume_texts, **kwargs)
    
    def match_score(
        self,
        resume_embedding: np.ndarray,
        job_embedding: np.ndarray
    ) -> float:
        """
        Calculate match score between resume and job
        
        Args:
            resume_embedding: Resume embedding vector
            job_embedding: Job embedding vector
        
        Returns:
            Similarity score between 0 and 1
        """
        similarity = self.similarity(resume_embedding, job_embedding)
        return float(similarity[0, 0])


# Convenience functions for backward compatibility
def get_embeddings() -> JobEmbeddings:
    """Get the singleton embeddings instance"""
    return JobEmbeddings()


def encode_text(text: Union[str, List[str]], **kwargs) -> np.ndarray:
    """Convenience function to encode text"""
    return get_embeddings().encode(text, **kwargs)


def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Convenience function to calculate similarity"""
    embeddings = get_embeddings()
    sim = embeddings.similarity(embedding1, embedding2)
    return float(sim[0, 0])


# Self-test
if __name__ == "__main__":
    print("Testing JobBERT-v2 embeddings...")
    
    # Test job posting
    job_text = """
    Senior Software Engineer
    We are looking for a Python developer with 5+ years of experience.
    Must have experience with Django, REST APIs, and cloud platforms.
    """
    
    # Test resume
    resume_text = """
    John Doe - Software Engineer
    5 years of experience in Python development.
    Skilled in Django, FastAPI, AWS, and building REST APIs.
    """
    
    # Get embeddings instance
    embeddings = JobEmbeddings()
    
    # Encode texts
    job_embedding = embeddings.encode_job(job_text)
    resume_embedding = embeddings.encode_resume(resume_text)
    
    # Calculate match score
    score = embeddings.match_score(resume_embedding, job_embedding)
    
    print(f"\nJob-Resume Match Score: {score:.2%}")
    print(f"Embedding shape: {job_embedding.shape}")
    print(f"Model used: {embeddings.llm_name}")
    
    # Test batch encoding
    jobs = [job_text, "Data Scientist role with Python", "DevOps Engineer"]
    job_embeddings = embeddings.encode_batch_jobs(jobs)
    print(f"\nBatch encoding shape: {job_embeddings.shape}")
    
    print("\nJobBERT-v2 embeddings working correctly!")

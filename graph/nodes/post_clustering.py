"""
Section Clustering LangGraph Node - Optimized for llama.cpp with Qwen3-4B
Performs document segmentation using embeddings and clustering with LLM-based labeling.
"""

from __future__ import annotations
import os
import json
import re
import logging
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Set
from enum import Enum

import numpy as np
from tqdm import tqdm

# Embeddings - use centralized JobBERT-v2
try:
    from graph.embeddings import JobEmbeddings, get_embeddings
    USE_CENTRALIZED_EMBEDDINGS = True
except ImportError:
    from sentence_transformers import SentenceTransformer
    USE_CENTRALIZED_EMBEDDINGS = False

# Clustering
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    logging.info("HDBSCAN not available, will use AgglomerativeClustering")

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering

# Validation
from pydantic import BaseModel, Field, ValidationError, field_validator

# LLM client (OpenAI-compatible for llama.cpp)
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

# -------------------------
# Configuration
# -------------------------

class Config:
    """Centralized configuration optimized for Qwen3-4B via llama.cpp"""
    
    # Text processing
    MICROCHUNK_MAX_LEN: int = 550  # characters
    MICROCHUNK_OVERLAP: int = 60   # characters
    MAX_CLUSTER_CHARS: int = 2000  # Reduced for Qwen3-4B context efficiency
    LLM_MAX_TRIM_CHARS: int = 150  # Reduced for more conservative trimming
    
    # Embeddings and clustering
    EMBEDDING_MODEL: str = "TechWolf/JobBERT-v2"  # Job-specific embeddings for better clustering
    MIN_SIM_THRESHOLD: float = 0.60  # cosine similarity for merging
    KNN_K: int = 8
    MIN_CLUSTER_SIZE: int = 2
    HDBSCAN_MIN_SAMPLES: int = 2
    AGGLOM_LINKAGE: str = "average"
    
    # LLM settings for llama.cpp
    DEFAULT_LLM_MODEL: str = "qwen3-4b-instruct-2507-f16"  # Model name in llama.cpp
    LLM_TEMPERATURE: float = 0.1  # Lower for more deterministic output
    LLM_MAX_TOKENS: int = 10000  # Reasonable limit for JSON responses
    LLM_TOP_P: float = 0.95
    LLM_TOP_K: int = 40
    LLM_REPEAT_PENALTY: float = 1.1
    LLM_MAX_RETRIES: int = 3
    LLM_TIMEOUT: int = 30
    LLM_SEED: int = 42  # For reproducibility
    
    # llama.cpp server defaults
    DEFAULT_BASE_URL: str = "http://localhost:8000/v1"  # Standard llama.cpp port
    
    # Controlled vocabulary for section titles
    CONTROLLED_TITLES: Dict[str, List[str]] = {
        "role_overview": ["about the role", "overview", "who you are", "what you'll do", "summary", "position summary"],
        "requirements": ["requirements", "qualifications", "must have", "we're looking for", "what we need"],
        "responsibilities": ["responsibilities", "duties", "what you'll do", "key responsibilities", "your responsibilities"],
        "nice_to_have": ["nice to have", "preferred", "bonus points", "preferred qualifications", "plus"],
        "compensation": ["salary", "compensation", "pay", "benefits", "perks", "total rewards", "comp & benefits"],
        "company": ["about us", "about the company", "who we are", "our mission", "company overview"],
        "location_type": ["location", "remote policy", "work arrangement", "work hours", "office location", "where"],
        "application": ["how to apply", "application", "contact", "next steps", "interview process", "apply now"],
        "legal": ["eeo", "equal opportunity", "privacy", "notice", "disability accommodation", "legal", "disclaimer"]
    }
    
    # Title display mapping
    TITLE_DISPLAY_MAP: Dict[str, str] = {
        "role_overview": "Role Overview",
        "requirements": "Requirements",
        "responsibilities": "Responsibilities",
        "nice_to_have": "Nice to Have",
        "compensation": "Compensation & Benefits",
        "company": "About the Company",
        "location_type": "Work Arrangement",
        "application": "How to Apply",
        "legal": "Legal / EEO"
    }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create config from dictionary, using defaults for missing values"""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# -------------------------
# Data Models
# -------------------------

class PositionSpace(Enum):
    """Track whether positions refer to raw or cleaned text"""
    RAW = "raw"
    CLEANED = "cleaned"


@dataclass
class TextMapping:
    """Maintains bidirectional mapping between raw and cleaned text"""
    raw_text: str
    cleaned_text: str
    raw_to_cleaned: Dict[int, int]  # raw_pos -> cleaned_pos
    cleaned_to_raw: Dict[int, int]  # cleaned_pos -> raw_pos
    
    def to_raw(self, cleaned_pos: int) -> int:
        """Convert cleaned position to raw position"""
        if cleaned_pos in self.cleaned_to_raw:
            return self.cleaned_to_raw[cleaned_pos]
        # Binary search for nearest position
        cleaned_positions = sorted(self.cleaned_to_raw.keys())
        if not cleaned_positions:
            return cleaned_pos
        if cleaned_pos < cleaned_positions[0]:
            return self.cleaned_to_raw[cleaned_positions[0]]
        if cleaned_pos > cleaned_positions[-1]:
            return self.cleaned_to_raw[cleaned_positions[-1]]
        # Find nearest
        left, right = 0, len(cleaned_positions) - 1
        while left < right:
            mid = (left + right) // 2
            if cleaned_positions[mid] < cleaned_pos:
                left = mid + 1
            else:
                right = mid
        return self.cleaned_to_raw[cleaned_positions[left]]
    
    def to_cleaned(self, raw_pos: int) -> int:
        """Convert raw position to cleaned position"""
        if raw_pos in self.raw_to_cleaned:
            return self.raw_to_cleaned[raw_pos]
        # Similar binary search logic
        raw_positions = sorted(self.raw_to_cleaned.keys())
        if not raw_positions:
            return raw_pos
        if raw_pos < raw_positions[0]:
            return self.raw_to_cleaned[raw_positions[0]]
        if raw_pos > raw_positions[-1]:
            return self.raw_to_cleaned[raw_positions[-1]]
        # Find nearest
        left, right = 0, len(raw_positions) - 1
        while left < right:
            mid = (left + right) // 2
            if raw_positions[mid] < raw_pos:
                left = mid + 1
            else:
                right = mid
        return self.raw_to_cleaned[raw_positions[left]]


@dataclass
class Chunk:
    """Text chunk with position tracking"""
    id: int
    text: str
    start_cleaned: int  # Position in cleaned text
    end_cleaned: int    # Position in cleaned text
    start_raw: Optional[int] = None  # Position in raw text
    end_raw: Optional[int] = None    # Position in raw text
    embedding: Optional[np.ndarray] = None
    
    def set_raw_positions(self, text_mapping: TextMapping):
        """Set raw positions using text mapping"""
        self.start_raw = text_mapping.to_raw(self.start_cleaned)
        self.end_raw = text_mapping.to_raw(self.end_cleaned)


@dataclass
class Cluster:
    """Document cluster/section"""
    id: int
    chunk_ids: List[int]
    start_raw: int
    end_raw: int
    title: str
    title_key: Optional[str]
    summary: str
    confidence: float
    salient_cues: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMLabel(BaseModel):
    """Validated LLM response for cluster labeling"""
    title_key: Optional[str] = None
    title: str = Field(..., min_length=1, max_length=50)
    summary: str = Field(..., min_length=1, max_length=500)
    salient_cues: List[str] = Field(default=[], max_length=6)
    trim_suggestion: Dict[str, int] = Field(default={"trim_leading_chars": 0, "trim_trailing_chars": 0})
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    @field_validator('trim_suggestion')
    def validate_trim(cls, v, values):
        """Ensure trim values are within bounds"""
        max_trim = Config.LLM_MAX_TRIM_CHARS
        v["trim_leading_chars"] = max(0, min(max_trim, v.get("trim_leading_chars", 0)))
        v["trim_trailing_chars"] = max(0, min(max_trim, v.get("trim_trailing_chars", 0)))
        return v


# -------------------------
# Text Processing
# -------------------------

class TextProcessor:
    """Handles text normalization and chunking"""
    
    @staticmethod
    def normalize_whitespace(text: str) -> TextMapping:
        """
        Clean text while preserving structure and maintaining position mapping.
        Returns TextMapping object with bidirectional position mappings.
        """
        if not text:
            return TextMapping(text, "", {}, {})
        
        cleaned_chars = []
        raw_to_cleaned_map = {}
        cleaned_to_raw_map = {}
        
        bullet_chars = {'*', '•', '-', '·', '–', '—', '►', '»', '▪', '●', '◦'}
        prev_was_space = False
        cleaned_pos = 0
        
        for raw_pos, char in enumerate(text):
            # Skip carriage returns
            if char == '\r':
                continue
                
            # Convert tabs to spaces
            if char == '\t':
                char = ' '
            
            # Preserve newlines
            if char == '\n':
                cleaned_chars.append('\n')
                raw_to_cleaned_map[raw_pos] = cleaned_pos
                cleaned_to_raw_map[cleaned_pos] = raw_pos
                cleaned_pos += 1
                prev_was_space = False
                continue
            
            # Preserve bullet characters
            if char in bullet_chars:
                cleaned_chars.append(char)
                raw_to_cleaned_map[raw_pos] = cleaned_pos
                cleaned_to_raw_map[cleaned_pos] = raw_pos
                cleaned_pos += 1
                prev_was_space = False
                continue
            
            # Collapse multiple spaces
            if char.isspace():
                if not prev_was_space:
                    cleaned_chars.append(' ')
                    raw_to_cleaned_map[raw_pos] = cleaned_pos
                    cleaned_to_raw_map[cleaned_pos] = raw_pos
                    cleaned_pos += 1
                    prev_was_space = True
            else:
                cleaned_chars.append(char)
                raw_to_cleaned_map[raw_pos] = cleaned_pos
                cleaned_to_raw_map[cleaned_pos] = raw_pos
                cleaned_pos += 1
                prev_was_space = False
        
        cleaned_text = ''.join(cleaned_chars)
        
        # Collapse multiple blank lines to max 2
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Update mappings if text was further modified
        if len(cleaned_text) != len(cleaned_chars):
            # Rebuild mappings after regex substitution
            # This is complex, so we'll do a simpler approach:
            # Just ensure the final positions are valid
            final_cleaned_to_raw = {}
            for cleaned_pos in range(len(cleaned_text)):
                if cleaned_pos in cleaned_to_raw_map:
                    final_cleaned_to_raw[cleaned_pos] = cleaned_to_raw_map[cleaned_pos]
                else:
                    # Find nearest valid position
                    for offset in range(1, len(cleaned_text)):
                        if cleaned_pos - offset in cleaned_to_raw_map:
                            final_cleaned_to_raw[cleaned_pos] = cleaned_to_raw_map[cleaned_pos - offset]
                            break
                        if cleaned_pos + offset in cleaned_to_raw_map:
                            final_cleaned_to_raw[cleaned_pos] = cleaned_to_raw_map[cleaned_pos + offset]
                            break
            cleaned_to_raw_map = final_cleaned_to_raw
        
        return TextMapping(
            raw_text=text,
            cleaned_text=cleaned_text,
            raw_to_cleaned=raw_to_cleaned_map,
            cleaned_to_raw=cleaned_to_raw_map
        )
    
    @staticmethod
    def create_chunks(text: str, text_mapping: TextMapping, config: Config) -> List[Chunk]:
        """Create overlapping text chunks with position tracking"""
        chunks = []
        text_len = len(text)
        chunk_id = 0
        position = 0
        
        while position < text_len:
            # Determine chunk end
            chunk_end = min(position + config.MICROCHUNK_MAX_LEN, text_len)
            
            # Try to break at sentence or paragraph boundary
            if chunk_end < text_len:
                chunk_text = text[position:chunk_end]
                
                # Look for break points in the last 100 chars
                lookback_start = max(0, len(chunk_text) - 100)
                lookback = chunk_text[lookback_start:]
                
                # Find sentence boundaries
                sentence_breaks = list(re.finditer(r'[.!?]\s+|\n\n|\n', lookback))
                if sentence_breaks:
                    # Use the last sentence boundary found
                    last_break = sentence_breaks[-1]
                    chunk_end = position + lookback_start + last_break.end()
            
            # Extract chunk text
            chunk_text = text[position:chunk_end].strip()
            
            if chunk_text:  # Only create chunk if non-empty
                chunk = Chunk(
                    id=chunk_id,
                    text=chunk_text,
                    start_cleaned=position,
                    end_cleaned=chunk_end
                )
                chunk.set_raw_positions(text_mapping)
                chunks.append(chunk)
                chunk_id += 1
            
            # Move to next position with overlap
            if chunk_end >= text_len:
                break
            position = max(position + 1, chunk_end - config.MICROCHUNK_OVERLAP)
        
        return chunks
    
    @staticmethod
    def apply_trim_to_boundaries(
        start_cleaned: int,
        end_cleaned: int,
        text: str,
        trim_leading: int,
        trim_trailing: int
    ) -> Tuple[int, int]:
        """Apply trimming at word boundaries to avoid cutting mid-word"""
        # Apply leading trim
        new_start = start_cleaned + trim_leading
        if new_start > start_cleaned and new_start < len(text):
            # Find next word boundary
            while new_start < len(text) and new_start < end_cleaned and text[new_start].isalnum():
                new_start += 1
        
        # Apply trailing trim
        new_end = end_cleaned - trim_trailing
        if new_end < end_cleaned and new_end > 0:
            # Find previous word boundary
            while new_end > 0 and new_end > new_start and text[new_end - 1].isalnum():
                new_end -= 1
        
        # Ensure valid boundaries
        if new_end <= new_start:
            return start_cleaned, end_cleaned
        
        return new_start, new_end


# -------------------------
# Embeddings and Clustering
# -------------------------

class EmbeddingProcessor:
    """Handles embedding generation and normalization"""
    
    def __init__(self, llm_name: str = None):
        if USE_CENTRALIZED_EMBEDDINGS:
            # Use centralized JobBERT-v2 embeddings
            self.embeddings = get_embeddings()
            self.llm_name = self.embeddings.llm_name
        else:
            # Fallback to direct SentenceTransformer
            self.llm_name = llm_name or Config.EMBEDDING_MODEL
            self._model = None
    
    @property
    def model(self):
        """Get model instance"""
        if USE_CENTRALIZED_EMBEDDINGS:
            return self.embeddings.model
        else:
            if self._model is None:
                self._model = SentenceTransformer(self.llm_name)
            return self._model
    
    def embed_chunks(self, chunks: List[Chunk], batch_size: int = 32) -> np.ndarray:
        """Generate normalized embeddings for chunks"""
        if not chunks:
            return np.array([])
        
        texts = [chunk.text for chunk in chunks]
        
        if USE_CENTRALIZED_EMBEDDINGS:
            # Use centralized embeddings
            embeddings = self.embeddings.encode(
                texts,
                batch_size=batch_size,
                normalize=True
            )
        else:
            # Fallback to direct encoding
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=len(texts) > 50,
                normalize_embeddings=True
            )
        
        # Store in chunks and return as array
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding.astype(np.float32)
        
        return np.vstack([chunk.embedding for chunk in chunks])
    
    @staticmethod
    def verify_normalized(embeddings: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Verify that embeddings are normalized"""
        norms = np.linalg.norm(embeddings, axis=1)
        return np.allclose(norms, 1.0, atol=tolerance)


class ClusteringStrategy:
    """Handles different clustering strategies"""
    
    @staticmethod
    def cluster_with_hdbscan(
        embeddings: np.ndarray,
        min_cluster_size: int = 3,
        min_samples: int = 2
    ) -> np.ndarray:
        """Cluster using HDBSCAN"""
        if not HAS_HDBSCAN:
            raise ImportError("HDBSCAN not available")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',
            cluster_selection_epsilon=0.0,
            cluster_selection_method='eom'
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Convert noise points (-1) to unique singleton clusters
        max_label = labels.max() if len(labels) > 0 else -1
        next_label = max_label + 1
        
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = next_label
                next_label += 1
        
        return labels
    
    @staticmethod
    def cluster_with_agglomerative(
        embeddings: np.ndarray,
        distance_threshold: float = 0.4,
        min_cluster_size: int = 2
    ) -> np.ndarray:
        """Cluster using Agglomerative Clustering"""
        from sklearn.metrics import pairwise_distances
        
        # Compute cosine distance matrix
        distance_matrix = pairwise_distances(embeddings, metric='cosine')
        
        # Perform clustering with precomputed distance matrix
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric='precomputed',
            linkage='average'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Post-process: enforce minimum cluster size
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Convert small clusters to singletons
        max_label = labels.max()
        next_label = max_label + 1
        
        for i in range(len(labels)):
            if label_counts[labels[i]] < min_cluster_size:
                labels[i] = next_label
                next_label += 1
        
        return labels
    
    @staticmethod
    def merge_adjacent_clusters(
        clusters: Dict[int, List[int]],
        chunks: List[Chunk],
        embeddings: np.ndarray,
        similarity_threshold: float = 0.6
    ) -> Dict[int, List[int]]:
        """Merge adjacent clusters if they are similar enough"""
        if not clusters:
            return clusters
        
        # Calculate cluster spans and centroids
        cluster_info = {}
        for cluster_id, chunk_ids in clusters.items():
            chunk_embeddings = embeddings[chunk_ids]
            centroid = np.mean(chunk_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize
            
            starts = [chunks[i].start_cleaned for i in chunk_ids]
            ends = [chunks[i].end_cleaned for i in chunk_ids]
            
            cluster_info[cluster_id] = {
                'centroid': centroid,
                'start': min(starts),
                'end': max(ends),
                'chunk_ids': chunk_ids
            }
        
        # Sort clusters by start position
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['start'])
        
        # Merge adjacent similar clusters
        merged = {}
        i = 0
        while i < len(sorted_clusters):
            current_id, current_info = sorted_clusters[i]
            merged[current_id] = current_info['chunk_ids'].copy()
            
            # Try to merge with next clusters
            j = i + 1
            while j < len(sorted_clusters):
                next_id, next_info = sorted_clusters[j]
                
                # Check if adjacent (with small gap tolerance)
                gap = next_info['start'] - current_info['end']
                if gap > 50:  # Max gap in characters
                    break
                
                # Check similarity
                similarity = np.dot(current_info['centroid'], next_info['centroid'])
                
                if similarity >= similarity_threshold:
                    # Merge
                    merged[current_id].extend(next_info['chunk_ids'])
                    
                    # Update current cluster info
                    new_embeddings = embeddings[merged[current_id]]
                    current_info['centroid'] = np.mean(new_embeddings, axis=0)
                    current_info['centroid'] = current_info['centroid'] / np.linalg.norm(current_info['centroid'])
                    current_info['end'] = max(current_info['end'], next_info['end'])
                    
                    # Remove merged cluster from list
                    sorted_clusters.pop(j)
                else:
                    j += 1
            
            i += 1
        
        # Sort chunk IDs within each cluster
        for cluster_id in merged:
            merged[cluster_id] = sorted(set(merged[cluster_id]))
        
        return merged


# -------------------------
# LLM Integration for Qwen3-4B via llama.cpp
# -------------------------

class QwenLLMProcessor:
    """Handles LLM-based cluster labeling optimized for Qwen3-4B via llama.cpp"""
    
    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        api_key: str = None,
        config: Config = None
    ):
        self.config = config or Config()
        self.model = model or os.getenv("LLM_MODEL", self.config.DEFAULT_LLM_MODEL)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", self.config.DEFAULT_BASE_URL)
        # llama.cpp doesn't need API key, but OpenAI client requires something
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "dummy-key")
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )
        
        logging.info(f"Initialized Qwen LLM processor with model: {self.model} at {self.base_url}")
    
    def _build_prompt_for_qwen(self, cluster_text: str) -> Tuple[str, str]:
        """Build prompts optimized for Qwen3-4B's instruction format"""
        
        # Qwen3 works best with clear, structured instructions
        system_prompt = """You are a document section classifier. Your task is to analyze text sections and provide structured JSON output.
Be concise, accurate, and only use information present in the provided text."""
        
        # Build controlled titles string for better parsing
        controlled_keys_str = ", ".join(f'"{key}"' for key in self.config.CONTROLLED_TITLES.keys())
        
        # Truncate cluster text if needed
        if len(cluster_text) > self.config.MAX_CLUSTER_CHARS:
            cluster_text = cluster_text[:self.config.MAX_CLUSTER_CHARS]
        
        # More structured prompt for Qwen3
        user_prompt = f"""Analyze the following text section and return a JSON object with these exact fields:

TEXT SECTION:
<<<
{cluster_text}
>>>

CONTROLLED TITLE KEYS (use one if applicable): [{controlled_keys_str}]

REQUIRED JSON FORMAT:
{{
  "title_key": <string or null>,     // Use controlled key if text matches, else null
  "title": <string>,                  // Human-readable title (1-5 words)
  "summary": <string>,                // Neutral 1-2 sentence summary
  "salient_cues": [<array>],         // Up to 6 key phrases from the text
  "trim_suggestion": {{
    "trim_leading_chars": <int>,     // 0-{self.config.LLM_MAX_TRIM_CHARS}
    "trim_trailing_chars": <int>     // 0-{self.config.LLM_MAX_TRIM_CHARS}
  }},
  "confidence": <float>               // 0.0-1.0 match confidence
}}

Respond with ONLY the JSON object, no additional text or markdown."""
        
        return system_prompt, user_prompt
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call Qwen via llama.cpp with retry logic"""
        try:
            # Build request with Qwen-optimized parameters
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS,
                top_p=self.config.LLM_TOP_P,
                # llama.cpp specific parameters (passed as extra_body)
                extra_body={
                    "top_k": self.config.LLM_TOP_K,
                    "repeat_penalty": self.config.LLM_REPEAT_PENALTY,
                    "seed": self.config.LLM_SEED,
                    "stop": ["```", "\n\n\n"],  # Stop sequences to prevent rambling
                    "stream": False
                },
                timeout=self.config.LLM_TIMEOUT
            )
            
            content = response.choices[0].message.content
            logging.debug(f"LLM response: {content[:200]}...")
            return content
            
        except Exception as e:
            logging.error(f"LLM call failed: {e}")
            raise
    
    def _parse_qwen_response(self, response: str) -> Dict[str, Any]:
        """Parse Qwen's response with robust error handling"""
        
        # Clean common issues with Qwen responses
        response = response.strip()
        
        # Remove markdown code blocks if present
        response = re.sub(r'^```(?:json)?\s*', '', response)
        response = re.sub(r'\s*```$', '', response)
        
        # Try direct JSON parsing
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            logging.debug(f"Initial JSON parse failed: {e}")
        
        # Extract JSON object from response
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            
            # Fix common JSON issues
            # Remove trailing commas
            json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
            # Fix Python-style booleans and None
            json_str = json_str.replace('None', 'null')
            json_str = json_str.replace('True', 'true')
            json_str = json_str.replace('False', 'false')
            # Fix single quotes (careful not to break strings with apostrophes)
            json_str = re.sub(r"'([^']*)'(?=\s*:)", r'"\1"', json_str)
            json_str = re.sub(r":\s*'([^']*)'", r': "\1"', json_str)
            
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logging.debug(f"Cleaned JSON parse failed: {e}")
        
        # Last resort: build a minimal valid response
        logging.warning("Failed to parse LLM response, using fallback")
        return {
            "title_key": None,
            "title": "Section",
            "summary": "Unable to parse LLM response",
            "salient_cues": [],
            "trim_suggestion": {"trim_leading_chars": 0, "trim_trailing_chars": 0},
            "confidence": 0.3
        }
    
    def label_cluster(self, cluster_text: str) -> LLMLabel:
        """Label a cluster with comprehensive error handling"""
        try:
            system_prompt, user_prompt = self._build_prompt_for_qwen(cluster_text)
            
            # Call LLM with retries
            response = self._call_llm(system_prompt, user_prompt)
            
            # Parse response
            data = self._parse_qwen_response(response)
            
            # Ensure all required fields exist with proper types
            cleaned_data = {
                "title_key": data.get("title_key"),
                "title": str(data.get("title", "Section"))[:50],  # Enforce max length
                "summary": str(data.get("summary", ""))[:500],
                "salient_cues": list(data.get("salient_cues", []))[:6],  # Max 6 items
                "trim_suggestion": {
                    "trim_leading_chars": min(self.config.LLM_MAX_TRIM_CHARS, 
                                             max(0, int(data.get("trim_suggestion", {}).get("trim_leading_chars", 0)))),
                    "trim_trailing_chars": min(self.config.LLM_MAX_TRIM_CHARS,
                                              max(0, int(data.get("trim_suggestion", {}).get("trim_trailing_chars", 0))))
                },
                "confidence": min(1.0, max(0.0, float(data.get("confidence", 0.5))))
            }
            
            # Validate with Pydantic
            return LLMLabel(**cleaned_data)
            
        except ValidationError as e:
            logging.warning(f"LLM response validation failed: {e}")
            # Return safe defaults
            return LLMLabel(
                title_key=None,
                title="Section",
                summary=cluster_text[:100] + "..." if len(cluster_text) > 100 else cluster_text,
                salient_cues=[],
                trim_suggestion={"trim_leading_chars": 0, "trim_trailing_chars": 0},
                confidence=0.3
            )
        except Exception as e:
            logging.error(f"Cluster labeling failed completely: {e}")
            # Return minimal defaults
            return LLMLabel(
                title_key=None,
                title="Section",
                summary="Error generating summary",
                salient_cues=[],
                trim_suggestion={"trim_leading_chars": 0, "trim_trailing_chars": 0},
                confidence=0.1
            )


# -------------------------
# Main Processing Pipeline
# -------------------------

class SectionClusteringPipeline:
    """Main pipeline for document section clustering"""
    
    def __init__(
        self,
        config: Config = None,
        embedding_processor: EmbeddingProcessor = None,
        llm_processor: QwenLLMProcessor = None
    ):
        self.config = config or Config()
        self.embedding_processor = embedding_processor or EmbeddingProcessor()
        self.llm_processor = llm_processor or QwenLLMProcessor(config=self.config)
        self.text_processor = TextProcessor()
    
    def process(
        self,
        raw_text: str,
        verbose: bool = False
    ) -> Tuple[List[Cluster], List[Chunk], Dict[str, Any]]:
        """
        Process raw text into labeled sections.
        
        Returns:
            - List of Cluster objects (sections)
            - List of Chunk objects (micro-chunks)
            - Dictionary of diagnostics
        """
        diagnostics = {
            "input_length": len(raw_text),
            "has_hdbscan": HAS_HDBSCAN,
            "errors": []
        }
        
        # Handle empty input
        if not raw_text or not raw_text.strip():
            diagnostics["status"] = "empty_input"
            return [], [], diagnostics
        
        try:
            # Step 1: Text normalization
            if verbose:
                logging.info("Normalizing text...")
            text_mapping = self.text_processor.normalize_whitespace(raw_text)
            diagnostics["cleaned_length"] = len(text_mapping.cleaned_text)
            
            # Step 2: Create chunks
            if verbose:
                logging.info("Creating text chunks...")
            chunks = self.text_processor.create_chunks(
                text_mapping.cleaned_text,
                text_mapping,
                self.config
            )
            diagnostics["num_chunks"] = len(chunks)
            
            if not chunks:
                diagnostics["status"] = "no_chunks_created"
                return [], chunks, diagnostics
            
            # Step 3: Generate embeddings
            if verbose:
                logging.info("Generating embeddings...")
            embeddings = self.embedding_processor.embed_chunks(chunks)
            diagnostics["embedding_shape"] = embeddings.shape
            diagnostics["embeddings_normalized"] = EmbeddingProcessor.verify_normalized(embeddings)
            
            # Step 4: Clustering
            if verbose:
                logging.info("Clustering chunks...")
            
            # Try HDBSCAN first if available
            if HAS_HDBSCAN:
                try:
                    labels = ClusteringStrategy.cluster_with_hdbscan(
                        embeddings,
                        min_cluster_size=self.config.MIN_CLUSTER_SIZE,
                        min_samples=self.config.HDBSCAN_MIN_SAMPLES
                    )
                    
                    # Check if too many singletons
                    unique_labels = np.unique(labels)
                    singleton_ratio = len(unique_labels) / len(labels)
                    
                    if singleton_ratio > 0.7:
                        if verbose:
                            logging.info(f"High singleton ratio ({singleton_ratio:.2f}), using agglomerative clustering")
                        labels = ClusteringStrategy.cluster_with_agglomerative(
                            embeddings,
                            distance_threshold=1 - self.config.MIN_SIM_THRESHOLD,
                            min_cluster_size=self.config.MIN_CLUSTER_SIZE
                        )
                        diagnostics["clustering_method"] = "agglomerative_fallback"
                    else:
                        diagnostics["clustering_method"] = "hdbscan"
                        
                except Exception as e:
                    logging.warning(f"HDBSCAN failed: {e}, using agglomerative clustering")
                    labels = ClusteringStrategy.cluster_with_agglomerative(
                        embeddings,
                        distance_threshold=1 - self.config.MIN_SIM_THRESHOLD,
                        min_cluster_size=self.config.MIN_CLUSTER_SIZE
                    )
                    diagnostics["clustering_method"] = "agglomerative_error_fallback"
                    diagnostics["errors"].append(str(e))
            else:
                labels = ClusteringStrategy.cluster_with_agglomerative(
                    embeddings,
                    distance_threshold=1 - self.config.MIN_SIM_THRESHOLD,
                    min_cluster_size=self.config.MIN_CLUSTER_SIZE
                )
                diagnostics["clustering_method"] = "agglomerative"
            
            # Build cluster dictionary
            clusters_dict = {}
            for chunk_idx, label in enumerate(labels):
                if label not in clusters_dict:
                    clusters_dict[label] = []
                clusters_dict[label].append(chunk_idx)
            
            diagnostics["num_initial_clusters"] = len(clusters_dict)
            
            # Step 5: Merge adjacent clusters
            if verbose:
                logging.info("Merging adjacent clusters...")
            clusters_dict = ClusteringStrategy.merge_adjacent_clusters(
                clusters_dict,
                chunks,
                embeddings,
                similarity_threshold=self.config.MIN_SIM_THRESHOLD
            )
            diagnostics["num_merged_clusters"] = len(clusters_dict)
            
            # Step 6: Label clusters with LLM
            if verbose:
                logging.info("Labeling clusters with Qwen3-4B...")
            
            clusters = []
            for cluster_id, chunk_ids in clusters_dict.items():
                # Get cluster text
                cluster_chunks = [chunks[i] for i in chunk_ids]
                cluster_text = ' '.join([c.text for c in cluster_chunks])
                
                # Get LLM label
                try:
                    label = self.llm_processor.label_cluster(cluster_text)
                except Exception as e:
                    logging.error(f"Failed to label cluster {cluster_id}: {e}")
                    diagnostics["errors"].append(f"Cluster {cluster_id}: {str(e)}")
                    label = LLMLabel(
                        title="Section",
                        summary="Unable to generate summary",
                        salient_cues=[],
                        trim_suggestion={"trim_leading_chars": 0, "trim_trailing_chars": 0},
                        confidence=0.1
                    )
                
                # Calculate boundaries with trimming
                first_chunk = cluster_chunks[0]
                last_chunk = cluster_chunks[-1]
                
                start_cleaned, end_cleaned = self.text_processor.apply_trim_to_boundaries(
                    first_chunk.start_cleaned,
                    last_chunk.end_cleaned,
                    text_mapping.cleaned_text,
                    label.trim_suggestion.get("trim_leading_chars", 0),
                    label.trim_suggestion.get("trim_trailing_chars", 0)
                )
                
                # Convert to raw positions
                start_raw = text_mapping.to_raw(start_cleaned)
                end_raw = text_mapping.to_raw(end_cleaned)
                
                # Map title key to display title
                display_title = label.title
                if label.title_key and label.title_key in self.config.TITLE_DISPLAY_MAP:
                    display_title = self.config.TITLE_DISPLAY_MAP[label.title_key]
                
                # Create cluster object
                cluster = Cluster(
                    id=cluster_id,
                    chunk_ids=chunk_ids,
                    start_raw=start_raw,
                    end_raw=end_raw,
                    title=display_title,
                    title_key=label.title_key,
                    summary=label.summary,
                    confidence=label.confidence,
                    salient_cues=label.salient_cues,
                    metadata={
                        "original_title": label.title,
                        "num_chunks": len(chunk_ids),
                        "cluster_text_length": len(cluster_text)
                    }
                )
                clusters.append(cluster)
            
            # Sort clusters by position
            clusters.sort(key=lambda c: c.start_raw)
            
            diagnostics["num_final_sections"] = len(clusters)
            diagnostics["status"] = "success"
            
            return clusters, chunks, diagnostics
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            diagnostics["status"] = "error"
            diagnostics["errors"].append(str(e))
            return [], [], diagnostics


# -------------------------
# LangGraph Node Interface
# -------------------------

def section_clustering_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph-compatible node for section clustering with Qwen3-4B.
    
    Expected state keys:
        - raw_text (str): Input text to process
        - config (dict, optional): Configuration overrides
        - verbose (bool, optional): Enable verbose logging
        - model_handles (dict, optional): Pre-initialized models
            - embedding_model: SentenceTransformer instance
            - llm_processor: QwenLLMProcessor instance
    
    Returns state with:
        - sections: List of section dictionaries
        - chunk_table: List of chunk dictionaries
        - diagnostics: Processing diagnostics
    """
    # Set up logging
    if state.get("verbose", False):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Get input text
    raw_text = state.get("raw_text", "")
    
    # Handle empty input
    if not raw_text or not raw_text.strip():
        state["sections"] = []
        state["chunk_table"] = []
        state["diagnostics"] = {
            "section_clustering": {
                "status": "empty_input",
                "input_length": 0
            }
        }
        return state
    
    # Load configuration
    config = Config()
    if "config" in state and isinstance(state["config"], dict):
        config = Config.from_dict(state["config"])
    
    # Initialize processors
    embedding_processor = None
    llm_processor = None
    
    # Check for pre-initialized models
    model_handles = state.get("model_handles", {})
    if "embedding_model" in model_handles:
        embedding_processor = EmbeddingProcessor()
        embedding_processor._model = model_handles["embedding_model"]
    
    if "llm_processor" in model_handles:
        llm_processor = model_handles["llm_processor"]
    else:
        # Create Qwen processor with any custom settings from state
        llm_config = state.get("llm_config", {})
        llm_processor = QwenLLMProcessor(
            model=llm_config.get("model"),
            base_url=llm_config.get("base_url"),
            api_key=llm_config.get("api_key"),
            config=config
        )
    
    # Create pipeline
    pipeline = SectionClusteringPipeline(
        config=config,
        embedding_processor=embedding_processor,
        llm_processor=llm_processor
    )
    
    # Process text
    try:
        clusters, chunks, diagnostics = pipeline.process(
            raw_text,
            verbose=state.get("verbose", False)
        )
        
        # Convert to dictionaries for state
        state["sections"] = [
            {
                "id": cluster.id,
                "title": cluster.title,
                "title_key": cluster.title_key,
                "summary": cluster.summary,
                "chunk_ids": cluster.chunk_ids,
                "start": cluster.start_raw,
                "end": cluster.end_raw,
                "confidence": cluster.confidence,
                "salient_cues": cluster.salient_cues,
                "metadata": cluster.metadata
            }
            for cluster in clusters
        ]
        
        state["chunk_table"] = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "start": chunk.start_raw,
                "end": chunk.end_raw,
                "start_cleaned": chunk.start_cleaned,
                "end_cleaned": chunk.end_cleaned
            }
            for chunk in chunks
        ]
        
        # Store diagnostics
        if "diagnostics" not in state:
            state["diagnostics"] = {}
        state["diagnostics"]["section_clustering"] = diagnostics
        
    except Exception as e:
        logging.error(f"Section clustering failed: {e}")
        state["sections"] = []
        state["chunk_table"] = []
        state["diagnostics"] = {
            "section_clustering": {
                "status": "error",
                "error": str(e)
            }
        }
    
    return state


# -------------------------
# Standalone Testing
# -------------------------

if __name__ == "__main__":
    # Example usage with llama.cpp
    sample_text = """
    Senior Data Engineer
    
    About Acme Analytics
    We are a Series B startup revolutionizing how companies understand their revenue data.
    Our platform processes billions of events daily and serves insights to Fortune 500 companies.
    
    The Role
    We're looking for a Senior Data Engineer to join our core infrastructure team.
    You'll be working on challenging problems at scale, building the foundation
    that powers our analytics platform.
    
    What You'll Do
    • Design and implement large-scale data pipelines processing terabytes of data daily
    • Build real-time streaming infrastructure using Kafka and Apache Flink
    • Optimize our Snowflake data warehouse for cost and performance
    • Develop and maintain our dbt models and data transformation layer
    • Partner with ML engineers to productionize machine learning features
    • Mentor junior engineers and contribute to architectural decisions
    
    Requirements
    • 5+ years of experience in data engineering or similar role
    • Expert-level Python and SQL skills
    • Deep experience with modern data stack (dbt, Snowflake, Fivetran, etc.)
    • Hands-on experience with streaming technologies (Kafka, Kinesis, Pub/Sub)
    • Strong understanding of data modeling and warehouse design
    • Experience with infrastructure as code (Terraform, CloudFormation)
    
    Nice to Have
    • Experience with Spark or similar distributed computing frameworks
    • Knowledge of machine learning pipelines and feature stores
    • Contributions to open-source data tools
    • Experience with data governance and compliance (GDPR, CCPA)
    
    Compensation & Benefits
    • Base salary: $150,000 - $180,000 depending on experience
    • Meaningful equity stake (0.1% - 0.3%)
    • Comprehensive health, dental, and vision insurance
    • $2,000 annual learning & development budget
    • Flexible PTO and parental leave
    • Home office stipend of $1,500
    
    Location & Work Style
    This role is fully remote within the United States. We have a distributed team
    across multiple time zones and prioritize async communication. Optional quarterly
    team gatherings in Denver for those who want to attend.
    
    How to Apply
    Please send your resume and a brief note about your most interesting data
    engineering project to careers@acme-analytics.com. We aim to respond to all
    applications within 48 hours.
    
    Equal Opportunity Statement
    Acme Analytics is an equal opportunity employer committed to building a diverse
    and inclusive team. We do not discriminate based on race, religion, color,
    national origin, gender, sexual orientation, age, marital status, veteran status,
    or disability status.
    """
    
    # Test with state dict (LangGraph style)
    # Make sure llama.cpp server is running with Qwen3-4B model loaded
    # Default: http://localhost:8080/v1
    test_state = {
        "raw_text": sample_text,
        "verbose": True,
        "config": {
            "MIN_SIM_THRESHOLD": 0.65,
            "LLM_TEMPERATURE": 0.1,  # Lower for more deterministic output
            "MAX_CLUSTER_CHARS": 2000  # Smaller context for Qwen3-4B
        },
        "llm_config": {
            "base_url": "http://localhost:8000/v1",  # llama.cpp default
            "model": "qwen3-4b-instruct-2507-f16"    # Your model name in llama.cpp
        }
    }
    
    print("Testing section clustering with Qwen3-4B via llama.cpp...")
    print(f"Ensure llama.cpp server is running at {test_state['llm_config']['base_url']}")
    print("-" * 60)
    
    result_state = section_clustering_node(test_state)
    
    print(f"\nFound {len(result_state['sections'])} sections:")
    for section in result_state["sections"]:
        print(f"\n[{section['id']}] {section['title']} (confidence: {section['confidence']:.2f})")
        print(f"  Summary: {section['summary']}")
        print(f"  Position: {section['start']}-{section['end']}")
        if section['salient_cues']:
            print(f"  Key phrases: {', '.join(section['salient_cues'])}")
    
    print(f"\nDiagnostics:")
    print(json.dumps(result_state["diagnostics"]["section_clustering"], indent=2))
    
    # Print sample configuration for llama.cpp server startup
    print("\n" + "=" * 60)
    print("To run llama.cpp server with Qwen3-4B:")
    print("./server -m models/qwen3-4b-instruct-2507-f16.gguf \\")
    print("  --host 0.0.0.0 \\")
    print("  --port 8000 \\")
    print("  --n-gpu-layers 35 \\")  # Adjust based on your GPU
    print("  --ctx-size 10000 \\")
    print("  --parallel 2")
    print("=" * 60)

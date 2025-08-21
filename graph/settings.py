"""
Consolidated settings management for the job matching system
Single source of truth for all configuration
"""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from enum import Enum
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Environment(Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


@dataclass
class DatabaseSettings:
    """Database configuration settings"""
    db_path: str = "data/lancedb"
    reset_if_exists: bool = False
    auto_recreate_on_schema_error: bool = True
    embedding_batch_size: int = 32
    max_search_results: int = 100
    cleanup_days: int = 30
    enable_versioning: bool = True


@dataclass
class CrawlerSettings:
    """Web crawler configuration"""
    headless: bool = True
    browser_timeout: int = 30000
    page_load_timeout: int = 60000
    crawl_delay_seconds: float = 1.5
    max_retries: int = 3
    retry_delay_seconds: float = 2.0
    default_max_jobs: int = 20
    max_jobs_per_session: int = 50
    job_card_selector: str = "div.EimVGf"
    show_more_selector: str = "button[aria-label*='Show full description']"


@dataclass
class LLMSettings:
    """LLM server configuration"""
    model_path: str = ""
    llm_name: str = "qwen3-4b-instruct-2507-f16"
    host: str = "0.0.0.0"
    port: int = 8000
    base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"))
    auto_start_server: bool = True
    stop_server_on_complete: bool = False
    server_startup_timeout: int = 60
    health_check_interval: int = 5
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 120
    context_size: int = 10000
    n_gpu_layers: int = 35
    threads: int = 12


@dataclass
class EmbeddingSettings:
    """Embedding model configuration"""
    llm_name: str = "TechWolf/JobBERT-v2"
    cache_embeddings: bool = True
    embedding_dimension: int = 768
    max_sequence_length: int = 512
    batch_size: int = 32


@dataclass
class PipelineSettings:
    """Pipeline execution settings"""
    output_directory: str = "data/pipeline_output"
    resume_directory: str = "data/resumes"
    jobs_directory: str = "data/jobs"
    cache_directory: str = "data/.cache"
    use_llm_matching: bool = True
    use_cache: bool = True
    verbose: bool = False
    save_intermediate_results: bool = True
    top_k_matches: int = 10
    min_match_score: float = 0.3
    parallel_processing: bool = True
    max_workers: int = 4


@dataclass
class LoggingSettings:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/graph.log"
    console_output: bool = True
    log_to_file: bool = False
    max_file_size_mb: int = 10
    backup_count: int = 5


@dataclass
class FeatureSettings:
    """Feature flags"""
    enable_caching: bool = True
    enable_monitoring: bool = False
    debug_mode: bool = False
    enable_telemetry: bool = False


@dataclass
class Settings:
    """Main settings container"""
    environment: Environment = Environment.DEVELOPMENT
    database: DatabaseSettings = field(default_factory=DatabaseSettings)
    crawler: CrawlerSettings = field(default_factory=CrawlerSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    pipeline: PipelineSettings = field(default_factory=PipelineSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    features: FeatureSettings = field(default_factory=FeatureSettings)
    
    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        """
        Load settings from consolidated configuration file
        Priority: Environment variables > Config file > Defaults
        """
        # Default config path
        if config_path is None:
            config_path = "config/settings.json"
        
        # Initialize with defaults
        settings = cls()
        
        # Load from config file if exists
        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file, "r") as f:
                    config_data = json.load(f)
                
                # Get environment from env var or config
                env_name = os.getenv("APP_ENVIRONMENT", 
                                    config_data.get("default", {}).get("environment", "development"))
                settings.environment = Environment(env_name)
                
                # Load default settings
                if "default" in config_data:
                    settings._apply_config(config_data["default"])
                
                # Apply environment-specific overrides
                if "environment_overrides" in config_data:
                    env_overrides = config_data["environment_overrides"].get(env_name, {})
                    if env_overrides:
                        settings._apply_config(env_overrides)
                
                logger.info(f"Loaded settings from {config_path} for environment: {env_name}")
                
            except Exception as e:
                logger.warning(f"Error loading config file {config_path}: {e}, using defaults")
        else:
            logger.info(f"Config file not found at {config_path}, using defaults")
        
        # Override with environment variables
        settings._apply_env_overrides()
        
        return settings
    
    def _apply_config(self, config_dict: Dict[str, Any]):
        """Apply configuration dictionary to settings"""
        if "database" in config_dict:
            for key, value in config_dict["database"].items():
                if hasattr(self.database, key):
                    setattr(self.database, key, value)
        
        if "crawler" in config_dict:
            for key, value in config_dict["crawler"].items():
                if hasattr(self.crawler, key):
                    setattr(self.crawler, key, value)
        
        if "llm" in config_dict:
            for key, value in config_dict["llm"].items():
                if hasattr(self.llm, key):
                    setattr(self.llm, key, value)
        
        if "embedding" in config_dict:
            for key, value in config_dict["embedding"].items():
                if hasattr(self.embedding, key):
                    setattr(self.embedding, key, value)
        
        if "pipeline" in config_dict:
            for key, value in config_dict["pipeline"].items():
                if hasattr(self.pipeline, key):
                    setattr(self.pipeline, key, value)
        
        if "logging" in config_dict:
            for key, value in config_dict["logging"].items():
                if hasattr(self.logging, key):
                    setattr(self.logging, key, value)
        
        if "features" in config_dict:
            for key, value in config_dict["features"].items():
                if hasattr(self.features, key):
                    setattr(self.features, key, value)
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        # Database settings
        if os.getenv("DB_PATH"):
            self.database.db_path = os.getenv("DB_PATH")
        if os.getenv("DB_RESET"):
            self.database.reset_if_exists = os.getenv("DB_RESET", "").lower() == "true"
        
        # Crawler settings
        if os.getenv("CRAWLER_HEADLESS"):
            self.crawler.headless = os.getenv("CRAWLER_HEADLESS", "").lower() == "true"
        if os.getenv("CRAWLER_MAX_JOBS"):
            self.crawler.default_max_jobs = int(os.getenv("CRAWLER_MAX_JOBS"))
        if os.getenv("CRAWL_DELAY_SECONDS"):
            self.crawler.crawl_delay_seconds = float(os.getenv("CRAWL_DELAY_SECONDS"))
        
        # LLM settings
        if os.getenv("LLM_MODEL_PATH"):
            self.llm.model_path = os.getenv("LLM_MODEL_PATH")
        if os.getenv("LLM_MODEL"):
            self.llm.llm_name = os.getenv("LLM_MODEL")
        if os.getenv("LLM_HOST"):
            self.llm.host = os.getenv("LLM_HOST")
        if os.getenv("LLM_PORT"):
            self.llm.port = int(os.getenv("LLM_PORT"))
        if os.getenv("OPENAI_BASE_URL"):
            self.llm.base_url = os.getenv("OPENAI_BASE_URL")
        if os.getenv("LLM_AUTO_START"):
            self.llm.auto_start_server = os.getenv("LLM_AUTO_START", "").lower() == "true"
        
        # Embedding settings
        if os.getenv("EMBEDDING_MODEL"):
            self.embedding.llm_name = os.getenv("EMBEDDING_MODEL")
        
        # Pipeline settings
        if os.getenv("PIPELINE_OUTPUT_DIR"):
            self.pipeline.output_directory = os.getenv("PIPELINE_OUTPUT_DIR")
        
        # Logging settings
        if os.getenv("LOG_LEVEL"):
            self.logging.level = os.getenv("LOG_LEVEL")
        
        # Feature flags
        if os.getenv("DEBUG_MODE"):
            self.features.debug_mode = os.getenv("DEBUG_MODE", "").lower() == "true"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            "environment": self.environment.value,
            "database": self.database.__dict__,
            "crawler": self.crawler.__dict__,
            "llm": self.llm.__dict__,
            "embedding": self.embedding.__dict__,
            "pipeline": self.pipeline.__dict__,
            "logging": self.logging.__dict__,
            "features": self.features.__dict__
        }
    
    def save(self, path: str):
        """Save current settings to file"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Settings saved to {path}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings(reload: bool = False) -> Settings:
    """
    Get the global settings instance
    
    Args:
        reload: Force reload settings from file
    
    Returns:
        Settings instance
    """
    global _settings
    
    if _settings is None or reload:
        _settings = Settings.load()
    
    return _settings


def reset_settings():
    """Reset the global settings instance"""
    global _settings
    _settings = None

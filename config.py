"""
Configuration settings for Kinetics Technical Assistant
"""

import os
from dotenv import load_dotenv
import logging

# Load environment variables from .env file if present
load_dotenv()

class Config:
    # === API Keys ===
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    GROK_API_KEY = os.getenv("GROK_API_KEY", "")  # For xAI Grok model
    BRAVE_SEARCH_API_KEY = os.getenv("BRAVE_SEARCH_API_KEY", "")
    
    # === Azure Services ===
    AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    CONTAINER_NAME = os.getenv("CONTAINER_NAME", "kinetics-ta")
    AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY", "")
    AZURE_SEARCH_SERVICE = os.getenv("AZURE_SEARCH_SERVICE", "")
    AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX", "kinetics-index")
    
    # === Qdrant Vector Database ===
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
    
    # === CrewAI Settings ===
    ENABLE_CREWAI = os.getenv("ENABLE_CREWAI", "True").lower() == "true"
    ENABLE_AB_TESTING = os.getenv("ENABLE_AB_TESTING", "False").lower() == "true"
    CREWAI_CACHE_TTL = int(os.getenv("CREWAI_CACHE_TTL", "24"))  # Cache time-to-live in hours
    
    # === File Paths ===
    LOG_DIR = os.getenv("LOG_DIR", "G:/APPS/Technical Assistant")
    CACHE_DIR = os.getenv("CACHE_DIR", "G:/APPS/Technical Assistant/cache")
    
    # === Model Settings ===
    DEFAULT_RESEARCH_MODEL = os.getenv("DEFAULT_RESEARCH_MODEL", "claude-3-5-sonnet-20241022")
    DEFAULT_DOCUMENT_MODEL = os.getenv("DEFAULT_DOCUMENT_MODEL", "gpt-4o")
    DEFAULT_EXPERT_MODEL = os.getenv("DEFAULT_EXPERT_MODEL", "grok-2-latest")
    
    # === Logging Settings ===
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # === Performance Settings ===
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "3"))
    SEARCH_TIMEOUT = int(os.getenv("SEARCH_TIMEOUT", "10"))  # Seconds
    MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "60"))  # Seconds
    
    # === Available AI Models ===
    AVAILABLE_MODELS = [
        "claude-3-5-sonnet-20240620",  # Default Claude model
        "claude-3-5-sonnet-20241022",  # Claude 3.5 Sonnet
        "claude-3-opus-20240229",      # Claude 3 Opus
        "gpt-4o-mini",                 # OpenAI GPT-4 Turbo Mini
        "gpt-4o",                      # OpenAI GPT-4o
        "grok-2-latest"                # xAI Grok-2
    ]
    DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "grok-2-latest")

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=os.path.join(Config.LOG_DIR, 'technical_assistant.log'),
    filemode='a'
)

# Set warnings level for specific libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
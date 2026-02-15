"""
Configuration module for Sales AI Assistant.
Centralizes all settings, paths, and constants.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "sales_ai.db"

# LLM Configuration
# Supports Ollama (recommended) or HuggingFace transformers
LLM_CONFIG = {
    "provider": "ollama",  # Options: "ollama", "transformers", "llama_cpp"
    "model": "llama3.2",   # Ollama model name (3B params, fast on CPU)
    "temperature": 0.3,    # Low temp for structured output
    "max_tokens": 1024,
    "timeout": 30,
    "gpu": True,           # Enable GPU acceleration
    "gpu_layers": -1,      # -1 = all layers on GPU (Ollama/llama.cpp)
    "cuda_device": 1,      # CUDA device ID for multi-GPU systems
}

# Alternative: HuggingFace local model with GPU
# LLM_CONFIG = {
#     "provider": "transformers",
#     "model": "microsoft/DialoGPT-medium",
#     "temperature": 0.3,
#     "max_tokens": 512,
#     "gpu": True,
#     "cuda_device": 0,
#     "quantization": "4bit",  # Options: None, "8bit", "4bit"
# }

# Alternative: llama.cpp with GPU (fastest local option)
# LLM_CONFIG = {
#     "provider": "llama_cpp",
#     "model_path": "./models/llama-3.2-3b-q4_0.gguf",
#     "temperature": 0.3,
#     "max_tokens": 1024,
#     "gpu": True,
#     "n_gpu_layers": -1,  # -1 = offload all to GPU
#     "n_ctx": 4096,
# }

# Deal stages for validation
VALID_DEAL_STAGES = [
    "prospecting",
    "qualification",
    "proposal",
    "negotiation",
    "closed_won",
    "closed_lost",
    "nurture"
]

# Interest levels for validation
VALID_INTEREST_LEVELS = [
    "hot",      # Ready to buy immediately
    "warm",     # Interested, needs nurturing
    "cold",     # Low interest
    "neutral"   # Unclear/assessing
]

# Date format for parsing
DATE_FORMAT = "%Y-%m-%d"

# Database schema version for migrations
DB_VERSION = "1.0.0"

# Duplicate detection threshold (0-100)
DUPLICATE_SIMILARITY_THRESHOLD = 85  # Name similarity percentage

def get_db_connection_string() -> str:
    """Return SQLite connection string."""
    return f"sqlite:///{DB_PATH}"

def ensure_directories() -> None:
    """Ensure required directories exist."""
    BASE_DIR.mkdir(exist_ok=True)

    # Create models directory for llama.cpp
    models_dir = BASE_DIR / "models"
    models_dir.mkdir(exist_ok=True)

def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and return status."""
    gpu_info = {
        "available": False,
        "type": None,
        "devices": 0,
        "device_names": []
    }

    # Check CUDA (NVIDIA)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["type"] = "cuda"
            gpu_info["devices"] = torch.cuda.device_count()
            gpu_info["device_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            return gpu_info
    except ImportError:
        pass

    # Check Metal (Apple Silicon)
    try:
        import torch
        if torch.backends.mps.is_available():
            gpu_info["available"] = True
            gpu_info["type"] = "mps"
            gpu_info["devices"] = 1
            gpu_info["device_names"] = ["Apple Metal (MPS)"]
            return gpu_info
    except:
        pass

    # Check ROCm (AMD)
    try:
        import torch
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            gpu_info["available"] = True
            gpu_info["type"] = "rocm"
            gpu_info["devices"] = torch.cuda.device_count()
            gpu_info["device_names"] = ["AMD ROCm"]
            return gpu_info
    except:
        pass

    return gpu_info
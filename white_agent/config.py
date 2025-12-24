"""
White Agent Configuration

Single source of truth for all White Agent environment variables.
Import from here instead of using os.environ.get() directly.
"""
import os

# Model Configuration
MODEL = os.environ.get("MODEL", os.environ.get("GPT5_MODEL", "gpt-5.1"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", os.environ.get("GPT4V_TEMPERATURE", "1.0")))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))

# Observation/Action Configuration
# For Qwen models, set OSWORLD_OBS_TYPE=screenshot (pure visual grounding)
OBSERVATION_TYPE = os.environ.get("OSWORLD_OBS_TYPE", "screenshot_a11y_tree")
ACTION_SPACE = os.environ.get("ACTION_SPACE", "pyautogui")
MAX_TRAJECTORY_LENGTH = int(os.environ.get("MAX_TRAJECTORY_LENGTH", "3"))

# Server
WHITE_AGENT_HOST = os.environ.get("HOST", "localhost")
WHITE_AGENT_PORT = int(os.environ.get("PORT", "9002"))

# Cloud Run
CLOUDRUN_HOST = os.environ.get("CLOUDRUN_HOST")


def get_agent_url() -> str:
    """Build the White Agent URL from environment or defaults."""
    agent_url = os.environ.get("AGENT_URL")
    if agent_url:
        return agent_url
    if CLOUDRUN_HOST:
        return f"https://{CLOUDRUN_HOST}"
    return f"http://{WHITE_AGENT_HOST}:{WHITE_AGENT_PORT}"


# API Keys - accessed via functions to support lazy loading
def get_azure_openai_api_key() -> str | None:
    return os.getenv("AZURE_OPENAI_API_KEY")


def get_azure_openai_endpoint() -> str | None:
    return os.getenv("AZURE_OPENAI_ENDPOINT")


def get_openai_base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "https://api.openai.com")


def get_reasoning_effort() -> str:
    return os.environ.get("GPT5_REASONING_EFFORT", "none")


def get_genai_api_key() -> str | None:
    return os.environ.get("GENAI_API_KEY")


def get_groq_api_key() -> str | None:
    return os.environ.get("GROQ_API_KEY")


def get_qwen3_vl_endpoint_url() -> str | None:
    """Get Qwen3-VL Vertex AI endpoint URL."""
    return os.environ.get("QWEN3_VL_ENDPOINT_URL")

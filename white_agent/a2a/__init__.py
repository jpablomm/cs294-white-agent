# White Agent - A2A Protocol Implementation
"""
AgentBeats-compliant A2A protocol server using the a2a SDK.

Note: The a2a SDK is only available when running in AgentBeats mode.
Install with: pip install a2a-sdk
"""

try:
    from .server import start_agent, PromptAgentExecutor
    __all__ = ["start_agent", "PromptAgentExecutor"]
except ImportError:
    # a2a SDK not installed - A2A mode not available
    start_agent = None
    PromptAgentExecutor = None
    __all__ = []

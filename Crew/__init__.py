"""
CrewAI integration module for Kinetics Technical Assistant
"""

from .agents import create_kinetics_crew
from .tasks import (
    create_kinetics_tasks,
    create_technical_specification_crew,
    create_installation_crew,
    select_appropriate_crew
)
from .integration import get_crew_response
from .caching import ResponseCache
from .memory import AgentMemory
from .feedback import AgentFeedback
from .error_handlers import with_error_handling, crew_fallback
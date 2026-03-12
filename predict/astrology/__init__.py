"""
Astrology Prediction Module

Provides lottery number predictions based on:
- 紫微斗數 (Zi Wei Dou Shu / Purple Star Astrology)
- 西洋星座 (Western Zodiac)

Uses Claude CLI (primary) with Gemini CLI fallback for AI-powered analysis.
"""

from .profiles import BirthProfileManager
from .gemini_client import GeminiAstrologyClient
from .claude_client import ClaudeAstrologyClient, RateLimitError

__all__ = [
    'BirthProfileManager',
    'GeminiAstrologyClient',
    'ClaudeAstrologyClient',
    'RateLimitError',
]

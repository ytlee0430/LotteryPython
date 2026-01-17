"""
Astrology Prediction Module

Provides lottery number predictions based on:
- 紫微斗數 (Zi Wei Dou Shu / Purple Star Astrology)
- 西洋星座 (Western Zodiac)

Uses Gemini CLI for AI-powered astrology analysis.
"""

from .profiles import BirthProfileManager
from .gemini_client import GeminiAstrologyClient

__all__ = ['BirthProfileManager', 'GeminiAstrologyClient']

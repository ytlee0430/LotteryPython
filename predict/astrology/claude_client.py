"""
Claude CLI Client for Astrology Predictions

Primary AI client for astrology-based lottery predictions.
Falls back to Gemini CLI when Claude hits rate limits.

Uses Claude CLI (claude -p) for non-interactive prompt execution.
"""

import subprocess
import time

from predict.astrology.gemini_client import GeminiAstrologyClient


class RateLimitError(Exception):
    """Raised when the AI provider returns a rate limit error."""
    pass


class ClaudeAstrologyClient(GeminiAstrologyClient):
    """Client for astrology predictions via Claude CLI.

    Inherits prompt generation, JSON parsing, and validation from
    GeminiAstrologyClient. Only overrides the CLI invocation layer.
    """

    AVAILABLE_MODELS = [
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-haiku-4-5-20251001",
    ]

    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, timeout: int = 180, model: str = None,
                 max_retries: int = 3, retry_delay: int = 3):
        self.timeout = timeout
        self.model = model or self.DEFAULT_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.claude_path = self._find_cli()

    def _find_cli(self) -> str:
        """Find the claude CLI executable path."""
        import shutil
        path = shutil.which('claude')
        if not path:
            raise RuntimeError("Claude CLI not found. Please install it first.")
        return path

    def _call_gemini(self, prompt: str) -> str:
        """Override: call Claude CLI instead of Gemini.

        Uses `claude -p` (print mode) for non-interactive execution.
        Raises RateLimitError when rate limit is detected.
        """
        try:
            cmd = [
                self.claude_path, "-p", prompt,
                "--output-format", "text",
                "--model", self.model,
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )

            output = result.stdout.strip()
            stderr = result.stderr.strip()

            # Detect rate limit errors
            combined = f"{output} {stderr}".lower()
            rate_limit_signals = [
                'rate limit', '429', 'too many requests',
                'overloaded', 'rate_limit',
            ]
            if any(s in combined for s in rate_limit_signals):
                raise RateLimitError(
                    f"Claude rate limit hit: {stderr or output[:200]}"
                )

            if result.returncode != 0 and not output:
                raise RuntimeError(
                    f"Claude CLI error (exit {result.returncode}): "
                    f"{stderr[:200]}"
                )

            return output
        except subprocess.TimeoutExpired:
            raise TimeoutError(
                f"Claude CLI timed out after {self.timeout}s"
            )
        except (RateLimitError, RuntimeError, TimeoutError):
            raise
        except Exception as e:
            raise RuntimeError(f"Claude CLI error: {e}")

"""
Math Validation Module
Uses SymPy to verify that LaTeX equations produced by agents are mathematically
parseable. This catches hallucinated or malformed expressions before they
break Manim's MathTex renderer.
"""

from __future__ import annotations

import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)

# Try to import sympy; degrade gracefully if not available
try:
    from sympy.parsing.latex import parse_latex
    from sympy import latex, sympify
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False
    logger.warning("SymPy not available — equation validation skipped.")


def _strip_delimiters(eq: str) -> str:
    """Strip $, $$, \\[, \\] delimiters from LaTeX strings."""
    eq = eq.strip()
    eq = re.sub(r"^\$\$|\$\$$", "", eq)   # $$ ... $$
    eq = re.sub(r"^\$|\$$", "", eq)        # $ ... $
    eq = re.sub(r"^\\\[|\\\]$", "", eq)   # \[ ... \]
    return eq.strip()


def validate_equation(latex_str: str) -> Tuple[bool, str]:
    """
    Attempt to parse a LaTeX equation with SymPy.
    Returns (is_valid, error_message).
    An empty error_message means success.
    """
    if not SYMPY_AVAILABLE:
        return True, ""  # Graceful degradation

    cleaned = _strip_delimiters(latex_str)
    if not cleaned:
        return False, "Empty equation string."

    try:
        parse_latex(cleaned)
        return True, ""
    except Exception as exc:
        return False, str(exc)


def validate_equations(equations: List[str]) -> List[Tuple[str, bool, str]]:
    """
    Validate a list of LaTeX equations.
    Returns a list of (original_string, is_valid, error_message) tuples.
    """
    results = []
    for eq in equations:
        valid, err = validate_equation(eq)
        if not valid:
            logger.warning("Invalid equation '%s': %s", eq[:60], err)
        results.append((eq, valid, err))
    return results


def filter_valid_equations(equations: List[str]) -> List[str]:
    """Return only the equations that pass SymPy validation."""
    validated = validate_equations(equations)
    valid = [eq for eq, ok, _ in validated if ok]
    invalid_count = len(equations) - len(valid)
    if invalid_count:
        logger.warning(
            "%d of %d equations failed validation and were removed.",
            invalid_count, len(equations)
        )
    return valid

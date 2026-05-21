"""
bio-evals - API
"""

__version__ = '0.1.0'


# Import API functions from submodules
from .testGeneProgramsConcordance import (
    testGeneProgramsConcordance,
    testGeneProgramsConcordanceStreaming,
)


# Re-export the functions to make them available directly from api
__all__ = [
    "testGeneProgramsConcordance",
    "testGeneProgramsConcordanceStreaming",
    # Add other API functions
]

# Version information
__version__ = '0.1.0'

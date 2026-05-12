"""Thin shim that imports _DensityKernel from placer.py for the test script."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from placer import _DensityKernel  # noqa: F401

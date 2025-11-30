import os
import sys

# Ensure the project root is on sys.path so tests can import packages like
# `tree` and `misc` when running directly from the repository.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Also add the project root (one level above the package) so absolute imports
# like ``import kl_clustering_analysis`` work without installing the package.
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

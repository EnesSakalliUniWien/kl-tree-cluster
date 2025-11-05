#!/usr/bin/env zsh
# UV Quick Reference - Copy/Paste these commands

# ============================================
# ACTIVATE ENVIRONMENT
# ============================================
cd /Users/berksakalli/Projects/kl-te-cluster
source .venv/bin/activate

# ============================================
# ADD UV TO PATH (do this once in ~/.zshrc)
# ============================================
export PATH="/Users/berksakalli/.local/bin:$PATH"

# ============================================
# INSTALL PACKAGES (ultra-fast!)
# ============================================
uv pip install numpy pandas scipy            # Core scientific
uv pip install networkx                      # Graph analysis
uv pip install -r requirements.txt           # All dependencies

# ============================================
# COMMON WORKFLOW
# ============================================
source .venv/bin/activate
jupyter notebook new_clustering_application.ipynb

# ============================================
# PROJECT STATS
# ============================================
# ✅ Virtual Environment: .venv/
# ✅ Total Packages: 116
# ✅ Install Time: 572ms (with uv)
# ✅ Python Version: 3.13.2
# ✅ Core Packages: numpy, pandas, scipy, sklearn, networkx

# ============================================
# UPDATE DEPENDENCIES
# ============================================
uv pip install --upgrade -r requirements.txt
uv pip freeze > requirements-lock.txt

# ============================================
# DEACTIVATE
# ============================================
deactivate

# ============================================
# FILES CREATED
# ============================================
# pyproject.toml         - Project metadata (PEP 517)
# requirements.txt       - Main dependencies
# UV_SETUP.md           - Detailed UV guide
# .venv/                - Virtual environment
# .envrc                - Environment setup script

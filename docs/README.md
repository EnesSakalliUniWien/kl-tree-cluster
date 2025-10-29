# KL-Clustering Analysis - Setup Guide

## Prerequisites
- Python 3.9 or higher
- `uv` package manager installed

### Install `uv` (if not already installed)
```bash
# On macOS with Homebrew
brew install uv

# Or with pip
pip install uv

# Or with curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup Instructions

### 1. Create Virtual Environment with `uv`
```bash
cd /Users/berksakalli/Projects/kl-te-cluster
uv venv
```

This creates a `.venv` directory with a Python virtual environment.

### 2. Activate the Virtual Environment
```bash
# On macOS/Linux
source .venv/bin/activate

# On Windows
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
# Install all core and dev dependencies
uv pip install -r requirements.txt

# Or install from pyproject.toml
uv pip install -e .

# Or install with optional dependencies (dev tools)
uv pip install -e ".[dev]"

# For SageMath support (optional, may take longer)
uv pip install -e ".[sage]"
```

### 4. Verify Installation
```bash
python -c "import numpy, pandas, scipy, matplotlib, sklearn, networkx; print('✓ All dependencies installed!')"
```

### 5. Start Jupyter Notebook
```bash
jupyter notebook new_clustering_application.ipynb
```

## Project Structure
```
kl-te-cluster/
├── pyproject.toml                          # Project configuration & dependencies
├── requirements.txt                        # Simple dependency list
├── README.md                               # This file
├── .venv/                                  # Virtual environment (auto-created)
└── new_clustering_application.ipynb        # Main analysis notebook
```

## Available Commands

### With activated venv:
```bash
# Run the notebook
jupyter notebook

# Run a specific notebook
jupyter notebook new_clustering_application.ipynb

# Run tests (if added)
pytest

# Format code with black
black .

# Lint code with pylint
pylint *.py
```

## Dependency Groups

### Core Dependencies
- **numpy**: Numerical computing
- **pandas**: Data manipulation
- **scipy**: Scientific computing (distributions, clustering)
- **matplotlib**: Plotting
- **scikit-learn**: Machine learning (MDS, metrics)
- **networkx**: Network/graph analysis
- **plotly**: Interactive visualizations

### Development Dependencies (`[dev]`)
- **jupyter**: Interactive notebooks
- **ipython**: Enhanced Python shell
- **notebook**: Jupyter web interface
- **black**: Code formatter
- **pylint**: Code linter
- **pytest**: Testing framework

### Optional Dependencies (`[sage]`)
- **sagemath**: Advanced mathematics (optional, large package)

## Quick Commands Reference

```bash
# Create environment
uv venv

# Activate
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install specific package
uv pip install networkx

# List installed packages
uv pip list

# Freeze dependencies
uv pip freeze > requirements-lock.txt

# Deactivate
deactivate
```

## Troubleshooting

### `uv: command not found`
Install uv: `pip install uv` or follow instructions at https://github.com/astral-sh/uv

### Virtual environment not activating
Try: `source .venv/bin/activate` (macOS/Linux)

### Jupyter kernel not found
```bash
python -m ipykernel install --user --name kl-cluster
jupyter notebook --list-kernels
```

### SageMath installation issues
SageMath is large and complex. Install without it initially:
```bash
uv pip install -e ".[dev]"  # Skip [sage]
```

## Next Steps
1. Activate the environment: `source .venv/bin/activate`
2. Start Jupyter: `jupyter notebook`
3. Run cells in `new_clustering_application.ipynb`
4. Install additional packages as needed: `uv pip install <package-name>`

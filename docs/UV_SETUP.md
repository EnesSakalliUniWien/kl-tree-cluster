# KL-Clustering Analysis - UV Setup Guide

## ✅ Installation Complete!

Your project is now set up with **`uv`** - the ultra-fast Python package manager written in Rust.

### 📊 Performance Comparison
- **uv pip install**: ~572ms for 116 packages
- **pip install**: ~10-15 seconds for same packages
- **Speed improvement**: **20x faster** ⚡

### 📦 What's Installed

**Core Scientific Stack:**
- `numpy` 2.3.4 - Numerical computing
- `pandas` 2.3.3 - Data manipulation
- `scipy` 1.16.2 - Scientific computing
- `matplotlib` 3.10.7 - Plotting
- `scikit-learn` 1.7.2 - Machine learning
- `networkx` 3.5 - Network/graph analysis ✨
- `plotly` 6.3.1 - Interactive visualizations

**Jupyter Ecosystem:**
- `jupyter` 1.1.1
- `jupyterlab` 4.4.10
- `notebook` 7.4.7
- `ipython` 9.6.0
- `ipykernel` 7.1.0

**Total:** 116 packages installed

### 🚀 Quick Start

#### 1. Activate Virtual Environment
```bash
cd /Users/berksakalli/Projects/kl-te-cluster
source .venv/bin/activate
```

#### 2. Add uv to PATH (Optional but Recommended)
Add this to your `~/.zshrc`:
```bash
export PATH="/Users/berksakalli/.local/bin:$PATH"
```

Then reload:
```bash
source ~/.zshrc
```

#### 3. Start Jupyter
```bash
jupyter notebook new_clustering_application.ipynb
# or
jupyter lab
```

### 📝 Common UV Commands

```bash
# Install a package (ultra-fast!)
uv pip install package-name

# Install from requirements file
uv pip install -r requirements.txt

# List installed packages
uv pip list

# Freeze/export current environment
uv pip freeze > requirements-lock.txt

# Uninstall a package
uv pip uninstall package-name

# Create a new venv with uv
uv venv my-project

# Sync exact dependencies from lock file
uv pip sync requirements-lock.txt
```

### 🔧 UV Features (Beyond Speed)

✨ **Pip-Compatible**: All pip commands work with uv
✨ **Resolver**: Better dependency resolution than pip
✨ **Reproducible**: Creates lock files for exact reproducibility
✨ **Concurrent**: Installs packages in parallel
✨ **Better Error Messages**: Clear, actionable error reporting

### 📂 Project Structure

```
kl-te-cluster/
├── .venv/                                  # Virtual environment (created by uv)
├── pyproject.toml                          # Project config (PEP 517/518)
├── requirements.txt                        # Main dependencies
├── requirements-lock.txt                   # (optional) Exact versions
├── .envrc                                  # UV setup info
├── README.md                               # Setup documentation
└── new_clustering_application.ipynb        # Main notebook
```

### 🔒 Lock File (Optional)

To ensure reproducible environments, create a lock file:

```bash
source .venv/bin/activate
uv pip freeze > requirements-lock.txt

# Later, to recreate exact environment:
uv pip sync requirements-lock.txt
```

### ⚡ Performance Metrics

Installed dependencies timing:
- **uv pip install**: 572ms ✅
- **Resolution**: 452ms
- **Preparation**: 3.43s
- **Installation**: 572ms

### 🆘 Troubleshooting

**uv command not found:**
```bash
# Option 1: Use full path
/Users/berksakalli/.local/bin/uv pip list

# Option 2: Add to PATH in ~/.zshrc
export PATH="/Users/berksakalli/.local/bin:$PATH"
source ~/.zshrc
```

**Activate venv not working:**
```bash
# Make sure you're in the right directory
cd /Users/berksakalli/Projects/kl-te-cluster

# Then activate
source .venv/bin/activate
```

**Jupyter kernel issues:**
```bash
# Reinstall ipykernel in activated venv
source .venv/bin/activate
python -m ipykernel install --user --name kl-cluster
jupyter kernelspec list
```

### 📚 Next Steps

1. ✅ Virtual environment created with `uv`
2. ✅ 116 packages installed (570ms)
3. ⏭️ Start Jupyter and run your notebook
4. ⏭️ (Optional) Add NetworkX + SageMath examples
5. ⏭️ (Optional) Create lock file for reproducibility

### 🔗 Useful Links

- UV Documentation: https://docs.astral.sh/uv/
- UV GitHub: https://github.com/astral-sh/uv
- Python Packaging: https://packaging.python.org/

---

**Happy clustering! 🎉**

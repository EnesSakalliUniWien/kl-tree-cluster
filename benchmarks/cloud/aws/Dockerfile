FROM python:3.11-slim

WORKDIR /app

ARG KL_TE_GIT_COMMIT=unknown
ARG KL_TE_GIT_BRANCH=unknown
ARG KL_TE_GIT_DIRTY=unknown
ENV KL_TE_GIT_COMMIT=${KL_TE_GIT_COMMIT}
ENV KL_TE_GIT_BRANCH=${KL_TE_GIT_BRANCH}
ENV KL_TE_GIT_DIRTY=${KL_TE_GIT_DIRTY}

COPY pyproject.toml README.md ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        awscli \
        "joblib>=1.3.0" \
        "matplotlib>=3.7.0" \
        "networkx>=3.1" \
        "numpy>=1.24.0,<2.4" \
        "pandas>=2.0.0" \
        "pydiffmap>=0.2.0.1" \
        "scikit-learn>=1.3.0" \
        "scipy>=1.10.0" \
        "statsmodels>=0.14.0" \
        "tabulate>=0.9.0"

COPY benchmarks ./benchmarks
COPY kl_clustering_analysis ./kl_clustering_analysis

ENTRYPOINT ["python", "-m"]

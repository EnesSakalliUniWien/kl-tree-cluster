FROM python:3.11-slim

WORKDIR /app

ARG TBS_GIT_COMMIT=unknown
ARG TBS_GIT_BRANCH=unknown
ARG TBS_GIT_DIRTY=unknown
ENV TBS_GIT_COMMIT=${TBS_GIT_COMMIT}
ENV TBS_GIT_BRANCH=${TBS_GIT_BRANCH}
ENV TBS_GIT_DIRTY=${TBS_GIT_DIRTY}

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
COPY tree_break_selection ./tree_break_selection

ENTRYPOINT ["python", "-m"]

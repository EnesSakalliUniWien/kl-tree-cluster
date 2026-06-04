FROM python:3.11-slim

WORKDIR /app

ARG KL_TE_GIT_COMMIT=unknown
ARG KL_TE_GIT_BRANCH=unknown
ENV KL_TE_GIT_COMMIT=${KL_TE_GIT_COMMIT}
ENV KL_TE_GIT_BRANCH=${KL_TE_GIT_BRANCH}

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY benchmarks ./benchmarks
COPY kl_clustering_analysis ./kl_clustering_analysis

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e ".[benchmark]" awscli

ENTRYPOINT ["python", "-m", "benchmarks.cloud.aws_selected_tail_equation_study"]

# AWS Batch Benchmark Diagnostics

The AWS image is a module-generic benchmark container. The Batch command chooses
which diagnostic module runs. These runners distribute validation workloads
only; they do not create production calibration fallbacks.

## Build

```bash
docker build -f benchmarks/cloud/aws/Dockerfile -t kl-te-benchmark-diagnostics:latest .
```

The image entrypoint is `python -m`, so job commands must begin with the Python
module path, for example `benchmarks.cloud.aws_selected_tail_equation_study` or
`benchmarks.cloud.aws_alpha_grid_search`.

## Local Alpha-Grid Smoke

Before submitting AWS work, run a one-case/two-shard smoke locally:

```bash
python -m benchmarks.cloud.aws_alpha_grid_search run-shard \
  --suite full \
  --case-names gauss_clear_small \
  --edge-alphas 0.001,0.003 \
  --sibling-alphas 0.01,0.03 \
  --output-dir benchmarks/results/aws_alpha_grid_smoke \
  --shard-count 2 \
  --shard-index 0

python -m benchmarks.cloud.aws_alpha_grid_search run-shard \
  --suite full \
  --case-names gauss_clear_small \
  --edge-alphas 0.001,0.003 \
  --sibling-alphas 0.01,0.03 \
  --output-dir benchmarks/results/aws_alpha_grid_smoke \
  --shard-count 2 \
  --shard-index 1

python -m benchmarks.cloud.aws_alpha_grid_search merge \
  --suite full \
  --case-names gauss_clear_small \
  --edge-alphas 0.001,0.003 \
  --sibling-alphas 0.01,0.03 \
  --output-dir benchmarks/results/aws_alpha_grid_smoke \
  --shard-count 2
```

## Run The Full Alpha Grid On AWS

Reauthenticate first if the AWS session has expired:

```bash
aws login
aws sts get-caller-identity
```

Build and push the image, then deploy the Batch stack:

```bash
aws ecr create-repository --repository-name kl-te-benchmark-diagnostics
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 \
  -f benchmarks/cloud/aws/Dockerfile \
  --build-arg KL_TE_GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg KL_TE_GIT_BRANCH=$(git branch --show-current) \
  -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kl-te-benchmark-diagnostics:latest \
  --push .
aws cloudformation deploy \
  --stack-name kl-te-benchmark-diagnostics-batch \
  --template-file benchmarks/cloud/aws/batch-stack.yml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ContainerImage=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kl-te-benchmark-diagnostics:latest \
      OutputBucket=YOUR_BUCKET \
      OutputPrefix=alpha-grid-full \
      SubnetIds=subnet-1,subnet-2 \
      SecurityGroupIds=sg-1
```

Submit one array job for the default 5x5 alpha grid. With `size=25`, each shard
owns one alpha pair over the full benchmark suite.

```bash
aws batch submit-job \
  --job-name kl-te-alpha-grid-shards \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --array-properties size=25 \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_alpha_grid_search",
      "run-shard",
      "--suite", "full",
      "--output-dir", "/tmp/alpha-grid-full",
      "--shard-count", "25",
      "--s3-uri", "s3://YOUR_BUCKET/alpha-grid-full"
    ]
  }'
```

After all shards finish, submit one merge job:

```bash
aws batch submit-job \
  --job-name kl-te-alpha-grid-merge \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_alpha_grid_search",
      "merge",
      "--suite", "full",
      "--output-dir", "/tmp/alpha-grid-full",
      "--shard-count", "25",
      "--s3-uri", "s3://YOUR_BUCKET/alpha-grid-full"
    ]
  }'
```

The merged alpha-grid outputs are:

- `merged/alpha_grid_summary.csv`: one row per alpha pair.
- `merged/alpha_grid_results.csv`: per-case KL benchmark rows for every alpha pair.
- `merged/aws_alpha_grid_manifest.json`: grid, shard count, git state, and output paths.

## Run Selected-Edge Type-I Geometry On AWS Batch

This diagnostic shards by replicate index. It estimates fixed-tree versus
same-data selected-tree edge behavior and records geometry covariates. It is
not a production calibration fallback.

Local container smoke:

```bash
docker run --rm kl-te-benchmark-diagnostics:local \
  benchmarks.cloud.aws_selected_edge_type1_geometry run-shard \
  --suite binary \
  --case-names binary_2clusters \
  --modes fixed_tree,selected_tree \
  --edge-alphas 0.0001,0.001 \
  --sibling-alpha 0.01 \
  --replicates 6 \
  --base-seed 20260604 \
  --output-dir /tmp/selected-edge-type1-pilot \
  --shard-count 2 \
  --shard-index 0
```

AWS pilot array:

```bash
aws batch submit-job \
  --job-name kl-te-selected-edge-type1-pilot \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --array-properties size=4 \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_selected_edge_type1_geometry",
      "run-shard",
      "--suite", "binary",
      "--case-names", "binary_2clusters,binary_low_noise_4c",
      "--modes", "fixed_tree,selected_tree",
      "--edge-alphas", "0.0001,0.001",
      "--sibling-alpha", "0.01",
      "--replicates", "40",
      "--base-seed", "20260604",
      "--output-dir", "/tmp/selected-edge-type1-pilot",
      "--shard-count", "4",
      "--s3-uri", "s3://YOUR_BUCKET/selected-edge-type1-pilot-20260604"
    ]
  }'
```

Merge:

```bash
aws batch submit-job \
  --job-name kl-te-selected-edge-type1-pilot-merge \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_selected_edge_type1_geometry",
      "merge",
      "--suite", "binary",
      "--case-names", "binary_2clusters,binary_low_noise_4c",
      "--modes", "fixed_tree,selected_tree",
      "--edge-alphas", "0.0001,0.001",
      "--sibling-alpha", "0.01",
      "--replicates", "40",
      "--base-seed", "20260604",
      "--output-dir", "/tmp/selected-edge-type1-pilot",
      "--shard-count", "4",
      "--s3-uri", "s3://YOUR_BUCKET/selected-edge-type1-pilot-20260604"
    ]
  }'
```

The merged selected-edge outputs are:

- `merged/selected_edge_geometry_edges.csv`
- `merged/selected_edge_geometry_siblings.csv`
- `merged/selected_edge_geometry_final.csv`
- `merged/aws_selected_edge_geometry_manifest.json`

## Run A Selected-Tail Shard Locally

```bash
python -m benchmarks.cloud.aws_selected_tail_equation_study run-shard \
  --output-dir raw/assets/benchmark-results/selected_tail_equation_cloud_run \
  --shard-count 20 \
  --replicates-per-shard 50 \
  --shard-index 0
```

## Run Selected-Tail On AWS Batch

Create the ECR repository, build and push the image, then deploy the Batch
stack:

```bash
aws ecr create-repository --repository-name kl-te-benchmark-diagnostics
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 \
  -f benchmarks/cloud/aws/Dockerfile \
  --build-arg KL_TE_GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg KL_TE_GIT_BRANCH=$(git branch --show-current) \
  -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kl-te-benchmark-diagnostics:latest \
  --push .
aws cloudformation deploy \
  --stack-name kl-te-benchmark-diagnostics-batch \
  --template-file benchmarks/cloud/aws/batch-stack.yml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ContainerImage=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kl-te-benchmark-diagnostics:latest \
      OutputBucket=YOUR_BUCKET \
      OutputPrefix=selected-tail-equation-cloud-run \
      SubnetIds=subnet-1,subnet-2 \
      SecurityGroupIds=sg-1
```

Submit the array job with size equal to `--shard-count`. AWS Batch provides
`AWS_BATCH_JOB_ARRAY_INDEX`, so `--shard-index` is not needed:

```bash
aws batch submit-job \
  --job-name kl-te-selected-tail-shards \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --array-properties size=20
```

After all shards finish, run one merge job against the same output directory:

```bash
aws batch submit-job \
  --job-name kl-te-selected-tail-merge \
  --job-queue kl-te-benchmark-diagnostics \
  --job-definition kl-te-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_selected_tail_equation_study",
      "merge",
      "--output-dir", "/tmp/selected-tail-equation-cloud-run",
      "--shard-count", "20",
      "--replicates-per-shard", "50",
      "--s3-uri", "s3://YOUR_BUCKET/selected-tail-equation-cloud-run"
    ]
  }'
```

The merge command downloads `s3://YOUR_BUCKET/selected-tail-equation-cloud-run/shards`
before recomputing combined outputs.

## Output Contract

- `shards/shard_XXXX/selected_geometry_records.csv`: shard row-level evidence.
- `shards/shard_XXXX/aws_shard_manifest.json`: shard seed, cases, git state,
  and output path.
- `merged/selected_geometry_records.csv`: combined row-level evidence with
  namespaced independent simulation ids.
- `merged/selected_ratio_tail_law.csv`: recomputed selected-tail support and
  held-out precision checks.
- `merged/aws_selected_tail_equation_study_manifest.json`: combined
  reproducibility manifest.

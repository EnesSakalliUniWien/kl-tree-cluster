# AWS Batch Benchmark Diagnostics

The AWS image is a module-generic benchmark container. The Batch command chooses
which diagnostic module runs. These runners distribute validation workloads
only; they do not create production calibration fallbacks.

## Build

```bash
docker build -f benchmarks/cloud/aws/Dockerfile -t tree-break-selection-benchmark-diagnostics:latest .
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
aws ecr create-repository --repository-name tree-break-selection-benchmark-diagnostics
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 \
  -f benchmarks/cloud/aws/Dockerfile \
  --build-arg TBS_GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg TBS_GIT_BRANCH=$(git branch --show-current) \
  -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/tree-break-selection-benchmark-diagnostics:latest \
  --push .
aws cloudformation deploy \
  --stack-name tree-break-selection-benchmark-diagnostics-batch \
  --template-file benchmarks/cloud/aws/batch-stack.yml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ContainerImage=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/tree-break-selection-benchmark-diagnostics:latest \
      OutputBucket=YOUR_BUCKET \
      OutputPrefix=alpha-grid-full \
      SubnetIds=subnet-1,subnet-2 \
      SecurityGroupIds=sg-1
```

Submit one array job for the default 5x5 alpha grid. With `size=25`, each shard
owns one alpha pair over the full benchmark suite.

```bash
aws batch submit-job \
  --job-name tree-break-selection-alpha-grid-shards \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
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
  --job-name tree-break-selection-alpha-grid-merge \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
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
- `merged/alpha_grid_results.csv`: per-case TBS benchmark rows for every alpha pair.
- `merged/aws_alpha_grid_manifest.json`: grid, shard count, git state, and output paths.

## Run KAK/Cosine Lens Linkage Alpha Sweeps On AWS

This diagnostic compares average, complete, and Ward-Euclidean trees for the
KAK/cosine subspace lenses. Ward is run on Euclidean lens coordinates; it is not
run on Hamming distances. Average and complete use Euclidean condensed
distances. Each AWS shard owns one or more `lens x linkage` groups and sweeps
all requested edge/sibling alpha pairs for those groups.

Local two-shard smoke:

```bash
python -m benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep run-shard \
  --output-dir benchmarks/results/aws_kak_lens_linkage_alpha_smoke \
  --shard-count 2 \
  --shard-index 0 \
  --edge-alphas 0.001 \
  --sibling-alphas 0.01 \
  --lenses raw_kak:binary:adaptive_modes_10_15 \
  --tree-linkage-methods average,complete,ward

python -m benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep run-shard \
  --output-dir benchmarks/results/aws_kak_lens_linkage_alpha_smoke \
  --shard-count 2 \
  --shard-index 1 \
  --edge-alphas 0.001 \
  --sibling-alphas 0.01 \
  --lenses raw_kak:binary:adaptive_modes_10_15 \
  --tree-linkage-methods average,complete,ward

python -m benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep merge \
  --output-dir benchmarks/results/aws_kak_lens_linkage_alpha_smoke \
  --shard-count 2 \
  --edge-alphas 0.001 \
  --sibling-alphas 0.01 \
  --lenses raw_kak:binary:adaptive_modes_10_15 \
  --tree-linkage-methods average,complete,ward
```

Submit the default big sweep as an 18-shard array job. The default grid contains
six lenses, three linkage methods, and a `5 x 5` alpha grid, so it produces `450`
summary rows before fail-closed gate rows are filtered by interpretation.

```bash
aws batch submit-job \
  --job-name tree-break-selection-kak-lens-linkage-alpha-shards \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
  --array-properties size=18 \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep",
      "run-shard",
      "--output-dir", "/tmp/kak-lens-linkage-alpha",
      "--shard-count", "18",
      "--s3-uri", "s3://YOUR_BUCKET/kak-lens-linkage-alpha"
    ]
  }'
```

After the shard jobs finish, merge the outputs:

```bash
aws batch submit-job \
  --job-name tree-break-selection-kak-lens-linkage-alpha-merge \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_kak_lens_linkage_alpha_sweep",
      "merge",
      "--output-dir", "/tmp/kak-lens-linkage-alpha",
      "--shard-count", "18",
      "--s3-uri", "s3://YOUR_BUCKET/kak-lens-linkage-alpha"
    ]
  }'
```

The merged outputs are:

- `merged/kak_lens_linkage_alpha_sweep_summary.csv`: one row per lens,
  linkage, and alpha pair.
- `merged/aws_kak_lens_linkage_alpha_sweep_manifest.json`: grid, lens list,
  linkage methods, shard count, git state, and output paths.

## Run The Tree-Strategy Semantic Panel On AWS

This diagnostic joins precomputed Julia result CSVs into one semantic panel with
tree strategy, lens family, linkage, alpha, cluster counts, context quality,
main-context refinement, feature-axis bridge, p-value continuity, and
radius/angle/action diagnostics. It does not rerun clustering.

Because the AWS image excludes `benchmarks/results`, first upload a
repository-shaped result bundle that contains the needed `benchmarks/results/...`
paths:

```bash
aws s3 sync benchmarks/results s3://YOUR_BUCKET/tree-strategy-input/benchmarks/results
```

Then run the panel builder:

```bash
aws batch submit-job \
  --job-name tree-break-selection-tree-strategy-semantic-panel \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_tree_strategy_semantic_panel",
      "run",
      "--input-s3-uri", "s3://YOUR_BUCKET/tree-strategy-input",
      "--output-dir", "/tmp/tree-strategy-semantic-panel",
      "--output-s3-uri", "s3://YOUR_BUCKET/tree-strategy-semantic-panel"
    ]
  }'
```

If the large KAK/cosine linkage alpha sweep has already been merged on S3, add
its repository-relative merged CSV path with `--alpha-summary-paths`, for
example:

```bash
      "--alpha-summary-paths",
      "benchmarks/results/00_current_20260427_blob_analysis/20260427_blob_analysis/17_aws_kak_lens_linkage_alpha_sweep_smoke_20260611/merged/kak_lens_linkage_alpha_sweep_summary.csv"
```

The panel outputs are:

- `tree_strategy_semantic_panel.csv`: the requested joined semantic table.
- `tree_strategy_semantic_panel_report.md`: counts and top refinement rows.
- `aws_tree_strategy_semantic_panel_manifest.json`: input/output paths, schema,
  row counts, semantic-role counts, and git state.

## Run Selected-Edge Type-I Geometry On AWS Batch

This diagnostic shards by replicate index. It estimates fixed-tree versus
same-data selected-tree edge behavior and records geometry covariates. It is
not a production calibration fallback.

Local container smoke:

```bash
docker run --rm tree-break-selection-benchmark-diagnostics:local \
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
  --job-name tree-break-selection-selected-edge-type1-pilot \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
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
  --job-name tree-break-selection-selected-edge-type1-pilot-merge \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
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

## Run Traversal Sibling-FDR Diagnostics On AWS Batch

This diagnostic shards by replicate index and runs exactly one FDR layer per
array job. Submit separate jobs for `synthetic_valid_p`, `fixed_tree_wald`,
`selected_tree_wald`, and `selected_tree_inflated` so each merged summary
describes one statistical object.

Local shard smoke:

```bash
python -m benchmarks.cloud.aws_traversal_sibling_fdr_null run-shard \
  --layer synthetic_valid_p \
  --suite binary \
  --case-names synthetic_balanced_binary_tree \
  --replicates 20 \
  --alpha 0.01 \
  --base-seed 20260604 \
  --output-dir benchmarks/results/aws_traversal_sibling_fdr_smoke \
  --shard-count 2 \
  --shard-index 0
```

AWS array example:

```bash
aws batch submit-job \
  --job-name tree-break-selection-traversal-sibling-fdr-shards \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
  --array-properties size=8 \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_traversal_sibling_fdr_null",
      "run-shard",
      "--layer", "selected_tree_wald",
      "--suite", "binary",
      "--case-names", "binary_2clusters,binary_low_noise_4c",
      "--replicates", "400",
      "--alpha", "0.01",
      "--edge-alpha", "0.001",
      "--base-seed", "20260604",
      "--output-dir", "/tmp/traversal-sibling-fdr",
      "--shard-count", "8",
      "--s3-uri", "s3://YOUR_BUCKET/traversal-sibling-fdr-selected-tree-wald"
    ]
  }'
```

Merge after all shards finish:

```bash
aws batch submit-job \
  --job-name tree-break-selection-traversal-sibling-fdr-merge \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
  --container-overrides '{
    "command": [
      "benchmarks.cloud.aws_traversal_sibling_fdr_null",
      "merge",
      "--layer", "selected_tree_wald",
      "--suite", "binary",
      "--case-names", "binary_2clusters,binary_low_noise_4c",
      "--replicates", "400",
      "--alpha", "0.01",
      "--edge-alpha", "0.001",
      "--base-seed", "20260604",
      "--output-dir", "/tmp/traversal-sibling-fdr",
      "--shard-count", "8",
      "--s3-uri", "s3://YOUR_BUCKET/traversal-sibling-fdr-selected-tree-wald"
    ]
  }'
```

Merged outputs:

- `merged/traversal_sibling_fdr_simulations.csv`
- `merged/traversal_sibling_fdr_summary.csv`
- `merged/aws_traversal_sibling_fdr_manifest.json`

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
aws ecr create-repository --repository-name tree-break-selection-benchmark-diagnostics
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 \
  -f benchmarks/cloud/aws/Dockerfile \
  --build-arg TBS_GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg TBS_GIT_BRANCH=$(git branch --show-current) \
  -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/tree-break-selection-benchmark-diagnostics:latest \
  --push .
aws cloudformation deploy \
  --stack-name tree-break-selection-benchmark-diagnostics-batch \
  --template-file benchmarks/cloud/aws/batch-stack.yml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ContainerImage=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/tree-break-selection-benchmark-diagnostics:latest \
      OutputBucket=YOUR_BUCKET \
      OutputPrefix=selected-tail-equation-cloud-run \
      SubnetIds=subnet-1,subnet-2 \
      SecurityGroupIds=sg-1
```

Submit the array job with size equal to `--shard-count`. AWS Batch provides
`AWS_BATCH_JOB_ARRAY_INDEX`, so `--shard-index` is not needed:

```bash
aws batch submit-job \
  --job-name tree-break-selection-selected-tail-shards \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
  --array-properties size=20
```

After all shards finish, run one merge job against the same output directory:

```bash
aws batch submit-job \
  --job-name tree-break-selection-selected-tail-merge \
  --job-queue tree-break-selection-benchmark-diagnostics \
  --job-definition tree-break-selection-benchmark-diagnostics \
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

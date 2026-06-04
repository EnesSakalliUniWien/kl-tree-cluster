# AWS Selected-Tail Equation Study

This runner distributes the selected-hierarchy geometry diagnostic across AWS
Batch array jobs. It does not create a production external calibration fallback.
Each shard writes row-level selected geometry records; the merge step recomputes
the combined selected-ratio tail-law and candidate-equation diagnostics from
those records.

## Build

```bash
docker build -f benchmarks/cloud/aws/Dockerfile -t kl-te-selected-tail:latest .
```

## Run A Shard Locally

```bash
python -m benchmarks.cloud.aws_selected_tail_equation_study run-shard \
  --output-dir raw/assets/benchmark-results/selected_tail_equation_cloud_run \
  --shard-count 20 \
  --replicates-per-shard 50 \
  --shard-index 0
```

## Run On AWS Batch

Create the ECR repository, build and push the image, then deploy the Batch
stack:

```bash
aws ecr create-repository --repository-name kl-te-selected-tail
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com
docker buildx build --platform linux/amd64 \
  -f benchmarks/cloud/aws/Dockerfile \
  --build-arg KL_TE_GIT_COMMIT=$(git rev-parse HEAD) \
  --build-arg KL_TE_GIT_BRANCH=$(git branch --show-current) \
  -t ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kl-te-selected-tail:latest \
  --push .
aws cloudformation deploy \
  --stack-name kl-te-selected-tail-batch \
  --template-file benchmarks/cloud/aws/batch-stack.yml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides \
      ContainerImage=ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/kl-te-selected-tail:latest \
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
  --job-queue kl-te-selected-tail \
  --job-definition kl-te-selected-tail \
  --array-properties size=20
```

After all shards finish, run one merge job against the same output directory:

```bash
aws batch submit-job \
  --job-name kl-te-selected-tail-merge \
  --job-queue kl-te-selected-tail \
  --job-definition kl-te-selected-tail \
  --container-overrides '{
    "command": [
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

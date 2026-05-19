# Glossary Update Runbook

Use this process whenever `glossary.xlsx` changes.

## 1. Replace The Glossary File

Place the updated glossary in the project root as:

```text
glossary.xlsx
```

## 2. Confirm Local Glossary Version

Run this from the project folder:

```bat
python -c "from pathlib import Path; import hashlib, datetime; p=Path('glossary.xlsx'); print(datetime.datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %I:%M %p local')); print(hashlib.sha256(p.read_bytes()).hexdigest()[:12]); print(p.stat().st_size)"
```

Record:

```text
Last updated
Version hash
File size
```

## 3. Choose A New Docker Tag

Example:

```text
v-glossary-YYYYMMDD-HHMM
```

## 4. Build And Push

Replace placeholders with actual AWS values.

```bat
docker build --no-cache -t <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPOSITORY_NAME>:<IMAGE_TAG> .
docker push <AWS_ACCOUNT_ID>.dkr.ecr.<AWS_REGION>.amazonaws.com/<ECR_REPOSITORY_NAME>:<IMAGE_TAG>
```

## 5. Create New ECS Task Revision

```text
ECS > Task definitions > <TASK_DEFINITION_FAMILY> > Create new revision
```

Change the container image to the new image tag.

## 6. Update ECS Service

```text
ECS > Clusters > <ECS_CLUSTER_NAME> > Services > <ECS_SERVICE_NAME> > Update service
```

Select the newest task revision and update.

## 7. Verify AWS Version

Open the deployed app URL and confirm the expected:

```text
Last updated
Glossary terms
Version hash
```

If the hash matches local, the deployed app is using the updated glossary.

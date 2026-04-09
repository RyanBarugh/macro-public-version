$ErrorActionPreference = "Stop"

$REGION = $env:AWS_REGION
$ACCOUNT = $env:AWS_ACCOUNT_ID
$FUNCTION = "macro-pipeline"
$REPO = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/$FUNCTION"

$versionFile = Join-Path (Get-Location) ".deploy-version"
if (Test-Path $versionFile) {
    $version = [int](Get-Content $versionFile) + 1
} else {
    $version = 20
}
$tag = "v$version"

Write-Host "`n=== Deploying $FUNCTION as $tag ===" -ForegroundColor Cyan

# 1. ECR login
Write-Host "Logging into ECR..." -ForegroundColor Yellow
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REPO.Split("/")[0]

# 2. Build
Write-Host "Building image..." -ForegroundColor Yellow
docker buildx build --platform linux/amd64 --provenance=false --output type=docker -t "${FUNCTION}:${tag}" .
if ($LASTEXITCODE -ne 0) { throw "Docker build failed" }

# 3. Tag + Push
Write-Host "Pushing to ECR..." -ForegroundColor Yellow
docker tag "${FUNCTION}:${tag}" "${REPO}:${tag}"
docker push "${REPO}:${tag}"
if ($LASTEXITCODE -ne 0) { throw "Docker push failed" }

# 4. Update Lambda
Write-Host "Updating Lambda..." -ForegroundColor Yellow
aws lambda wait function-updated --function-name $FUNCTION --region $REGION
aws lambda update-function-code --function-name $FUNCTION --image-uri "${REPO}:${tag}" --region $REGION
if ($LASTEXITCODE -ne 0) { throw "Lambda update failed" }

# 5. Save version
[System.IO.File]::WriteAllText($versionFile, "$version")

Write-Host "`n=== Deployed $tag ===" -ForegroundColor Green
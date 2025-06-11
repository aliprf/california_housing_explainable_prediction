#!/bin/bash
set -euo pipefail

# Usage: ./deploy.sh
# Requires:
# - gcloud SDK installed & logged in
# - Docker installed & running
# - kubectl installed
# - helm installed
# - service account JSON key as key-file.json (or adapt script)

# Function to fetch secrets from GCP Secret Manager
get_secret() {
  local secret_name=$1
  local project_id=$2
  gcloud secrets versions access latest --secret="$secret_name" --project="$project_id"
}

echo "Fetching PROJECT_ID secret..."
PROJECT_ID=$(get_secret "PROJECT_ID" "your-secrets-project")
if [[ -z "$PROJECT_ID" ]]; then
  echo "ERROR: PROJECT_ID secret missing."
  exit 1
fi

echo "Fetching other secrets..."
CLUSTER_NAME=$(get_secret "CLUSTER_NAME" "$PROJECT_ID")
DEPLOYMENT_NAME=$(get_secret "DEPLOYMENT_NAME" "$PROJECT_ID")
IMAGE_NAME=$(get_secret "IMAGE_NAME" "$PROJECT_ID")
NAMESPACE=$(get_secret "NAMESPACE" "$PROJECT_ID")
REGION=$(get_secret "REGION" "$PROJECT_ID")
REPO_NAME=$(get_secret "REPO_NAME" "$PROJECT_ID")

for var in CLUSTER_NAME DEPLOYMENT_NAME IMAGE_NAME NAMESPACE REGION REPO_NAME; do
  if [[ -z "${!var}" ]]; then
    echo "ERROR: $var secret missing."
    exit 1
  fi
done

TIMESTAMP=$(date +%Y%m%d%H%M%S)
IMAGE_FULL_PATH="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${TIMESTAMP}"

echo "Building Docker image: $IMAGE_FULL_PATH"
docker build -t "$IMAGE_FULL_PATH" .

echo "Authenticating Docker to Artifact Registry"
gcloud auth configure-docker "${REGION}-docker.pkg.dev"

echo "Pushing image"
docker push "$IMAGE_FULL_PATH"

echo "Fetching GKE cluster credentials"
gcloud container clusters get-credentials "$CLUSTER_NAME" --region "$REGION" --project "$PROJECT_ID"

echo "Creating namespace if missing"
if ! kubectl get namespace "$NAMESPACE" &>/dev/null; then
  kubectl create namespace "$NAMESPACE"
fi

echo "Creating Kubernetes secret for app secrets"
kubectl create secret generic app-secrets --namespace "$NAMESPACE" \
  --from-literal=DEPLOYMENT_NAME="$DEPLOYMENT_NAME" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "Creating image pull secret for Artifact Registry"
kubectl create secret docker-registry regcred \
  --docker-server="${REGION}-docker.pkg.dev" \
  --docker-username=_json_key \
  --docker-password="$(cat key-file.json)" \
  --docker-email=you@example.com \
  --namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

echo "Deploying Helm chart"
helm upgrade --install "$DEPLOYMENT_NAME" ./myapp-chart \
  --namespace "$NAMESPACE" \
  --set deploymentName="$DEPLOYMENT_NAME" \
  --set namespace="$NAMESPACE" \
  --set imageFullPath="$IMAGE_FULL_PATH" \
  --set projectId="$PROJECT_ID" \
  --set region="$REGION" \
  --set repoName="$REPO_NAME" \
  --set clusterName="$CLUSTER_NAME" \
  --set secrets.DEPLOYMENT_NAME="$DEPLOYMENT_NAME"

echo "Deployment completed successfully."

#!/bin/bash

# Configure GCP credentials for the project

set -e

echo "================================================"
echo "GCP Credentials Configuration"
echo "================================================"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed."
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

echo ""
echo "Current gcloud configuration:"
gcloud config list

# Login to GCP
echo ""
read -p "Do you want to authenticate with GCP? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Logging in to GCP..."
    gcloud auth login
fi

# Set application default credentials
echo ""
read -p "Do you want to set application default credentials? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Setting application default credentials..."
    gcloud auth application-default login
fi

# Set project
echo ""
read -p "Enter your GCP project ID: " project_id
if [ ! -z "$project_id" ]; then
    echo "Setting project to: $project_id"
    gcloud config set project $project_id
    echo "export GCP_PROJECT_ID=$project_id" >> ~/.bashrc
    echo "export GCP_PROJECT_ID=$project_id" >> ~/.zshrc
fi

# Set region
echo ""
read -p "Enter your preferred GCP region (default: us-central1): " region
region=${region:-us-central1}
echo "Setting region to: $region"
gcloud config set compute/region $region

echo ""
echo "================================================"
echo "GCP Configuration Complete!"
echo "================================================"
echo ""
echo "Current configuration:"
gcloud config list
echo ""

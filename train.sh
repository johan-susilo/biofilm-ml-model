#!/bin/bash

# Biofilm Prediction ML Model Training Script
# 
# This script trains the XGBoost and Random Forest models using Docker
# to ensure all dependencies are properly installed and configured.

set -e  # Exit on any error

echo "Biofilm Prediction Model Training"
echo "===================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not available in PATH"
    echo "Please install Docker and try again"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed or not available in PATH"
    echo "Please install docker-compose and try again"
    exit 1
fi

# Ensure we're in the right directory
if [[ ! -f "docker-compose.yml" ]]; then
    echo "Error: docker-compose.yml not found"
    echo "Please run this script from the software-igem directory"
    exit 1
fi

# Check if training data exists
if [[ ! -f "data/polished.csv" ]]; then
    echo "Error: Training data (data/polished.csv) not found"
    echo "Please ensure the training dataset is available"
    exit 1
fi

echo "Found training data: data/polished.csv"
echo "Setting up training environment..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Build the Docker image if needed
echo "Building Docker image..."
docker-compose build biofilm-api

echo "Starting model training..."
echo "   This may take several minutes depending on your hardware"
echo "   Training progress will be displayed below:"
echo ""

# Run the training using docker-compose with the training profile
docker-compose --profile training run --rm biofilm-trainer

# Check if models were created successfully
if [[ -f "ml-model/xgb_biofilm_model.json" ]] && [[ -f "ml-model/rf_uncertainty_model.joblib" ]]; then
    echo ""
    echo "Training completed successfully!"
    echo "Models saved:"
    echo "   - XGBoost model: ml-model/xgb_biofilm_model.json"
    echo "   - Random Forest model: ml-model/rf_uncertainty_model.joblib"
    echo ""
    echo "You can now start the API server with:"
    echo "   docker-compose up -d biofilm-api"
    echo ""
    echo "Or use the convenience script:"
    echo "   ./start.sh"
else
    echo ""
    echo "Training failed - models were not created"
    echo "Check the training logs above for error details"
    exit 1
fi

echo "Training complete! Your models are ready to use."

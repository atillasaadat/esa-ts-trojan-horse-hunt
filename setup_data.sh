#!/bin/bash

# Exit on error
set -e

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Ensure unzip is installed
echo "Checking for unzip..."
if ! command -v unzip &> /dev/null
then
    echo "unzip not found. Installing..."
    sudo apt-get update && sudo apt-get install -y unzip
fi

# Download the competition dataset
echo "Downloading dataset from Kaggle..."
kaggle competitions download -c trojan-horse-hunt-in-space

# Create data directory if it doesn't exist
mkdir -p ./data

# Unzip into ./data
echo "Unzipping dataset into ./data..."
unzip -o trojan-horse-hunt-in-space.zip -d ./data

# Delete the downloaded zip file
echo "Removing ZIP file..."
rm trojan-horse-hunt-in-space.zip

echo "Data setup complete!"

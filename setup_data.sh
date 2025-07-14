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
    echo "unzip not found. Attempting to install..."
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "Detected apt-based Linux. Installing unzip..."
            sudo apt-get update && sudo apt-get install -y unzip
        elif command -v dnf &> /dev/null; then
            echo "Detected Fedora-based Linux. Installing unzip..."
            sudo dnf install -y unzip
        else
            echo "Unsupported Linux distribution. Please install unzip manually."
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Detected macOS. Installing unzip..."
        brew install unzip || { echo "Homebrew not found. Please install unzip manually."; exit 1; }
    elif [[ "$OSTYPE" == "msys"* || "$OSTYPE" == "cygwin"* ]]; then
        echo "Detected Windows. Installing unzip..."
        choco install unzip || { echo "Chocolatey not found. Please install unzip manually."; exit 1; }
    else
        echo "Unsupported operating system. Please install unzip manually."
        exit 1
    fi
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

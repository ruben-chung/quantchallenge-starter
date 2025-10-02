#!/bin/bash

echo "Starting the environment setup..."
module purge
unset LD_LIBRARY_PATH
module load cudnn
module load cuda

# Create and activate virtual environment
python -m venv pytorch.venv
source pytorch.venv/bin/activate

# Upgrade pip and install packages
pip install --upgrade pip
pip install torch torchvision torchaudio matplotlib

echo "Done. Your environment is ready."


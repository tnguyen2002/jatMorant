#!/bin/bash

# Setup script for SFT training on GPU

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "conda is not installed. Please install miniconda or anaconda first."
    exit 1
fi

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml

# Activate environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sft-env

# Check if CUDA is available
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device count: {torch.cuda.device_count()}'); print(f'GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Download and prepare datasets (optional)
echo "Do you want to download and preprocess datasets? (y/n)"
read answer
if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
    echo "Preprocessing datasets..."
    python -c "from data import get_dataloaders; get_dataloaders(batch_size=4)"
fi

# Display help message
echo ""
echo "============================================================"
echo "Environment setup complete!"
echo "To run SFT training, use the following command:"
echo "python main.py --mode train --algorithm sft --batch_size 16 --learning_rate 2e-5 --num_epochs 3 --output_dir ./models/sft"
echo ""
echo "For faster training with larger batch sizes, you can use:"
echo "python main.py --mode train --algorithm sft --batch_size 32 --learning_rate 3e-5 --num_epochs 3 --output_dir ./models/sft"
echo ""
echo "To enable Weights & Biases tracking, add --use_wandb flag"
echo "============================================================" 
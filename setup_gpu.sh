#!/bin/bash

# Setup script for SFT training on GPU

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is not installed. Installing miniconda..."
    
    # Download Miniconda installer
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    chmod +x Miniconda3-latest-Linux-x86_64.sh
    
    # Install Miniconda
    ./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    
    # Add to PATH and initialize
    export PATH="$HOME/miniconda/bin:$PATH"
    
    # Initialize conda for bash
    $HOME/miniconda/bin/conda init bash
    
    # Source bashrc to apply changes
    source ~/.bashrc
    
    # Install necessary system packages
    echo "Installing system dependencies..."
    sudo apt update
    sudo apt install -y g++ make swig
    
    echo "Conda installation complete!"
fi

# Make sure conda commands are available in this script
export PATH="$HOME/miniconda/bin:$PATH"
source $(conda info --base)/etc/profile.d/conda.sh

# Create conda environment from environment.yml
echo "Creating conda environment from environment.yml..."
conda env create -f environment.yml || conda env update -f environment.yml

# Activate environment
echo "Activating conda environment..."
conda activate cs224r-env  # Using name from environment.yml

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
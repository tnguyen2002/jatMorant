#!/bin/bash

# Environment setup script for SFT training on g4dn.xlarge instance
# Optimized for Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.6.0 (Amazon Linux 2023)
# This script only sets up the environment and does not start training or dataset download

echo "=== Setting up environment for CS224R SFT on g4dn.xlarge ==="

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
    
    echo "Conda installation complete!"
else
    echo "Conda is already installed. Proceeding with setup..."
fi

# Create conda environment
echo "=== Creating conda environment ==="
conda create -n cs224r python=3.10 -y

# Activate environment
echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate cs224r

# Install required packages
echo "Installing dependencies..."
# First check if PyTorch with CUDA is already installed in the AMI
if python -c "import torch; exit(0) if torch.cuda.is_available() else exit(1)" 2>/dev/null; then
    echo "PyTorch with CUDA already installed, skipping PyTorch installation"
    # Install only the additional packages
    pip install transformers==4.35.0 datasets==2.14.5 wandb==0.16.0 \
        openai==1.3.0 bitsandbytes==0.41.0 peft==0.5.0 \
        evaluate==0.4.0 scikit-learn==1.3.1 tqdm pandas
else
    # Install PyTorch and all dependencies
    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
    pip install transformers==4.35.0 datasets==2.14.5 wandb==0.16.0 \
        openai==1.3.0 bitsandbytes==0.41.0 peft==0.5.0 \
        evaluate==0.4.0 scikit-learn==1.3.1 tqdm pandas
fi

# Verify GPU availability
echo "=== Checking CUDA and GPU availability ==="
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU device count: {torch.cuda.device_count()}'); print(f'GPU device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Display T4 GPU-optimized settings
echo "=== g4dn.xlarge T4 GPU Recommendations ==="
echo "Recommended settings for NVIDIA T4 (16GB) on g4dn.xlarge:"
echo "- Batch size: 16"
echo "- Learning rate: 2e-5"
echo "- Number of epochs: 3"

# Help message for next steps
echo ""
echo "=== Environment Setup Complete! ==="
echo ""
echo "To download and prepare datasets:"
echo "  python -c \"from data import get_dataloaders; get_dataloaders(batch_size=4)\""
echo ""
echo "To run SFT training:"
echo "  python main.py --mode train --algorithm sft --batch_size 16 --learning_rate 2e-5 --num_epochs 3 --output_dir ./models/sft"
echo ""
echo "To monitor GPU usage during training, run in a separate terminal:"
echo "  watch -n 1 nvidia-smi"
echo "" 
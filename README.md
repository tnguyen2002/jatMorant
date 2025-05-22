# Setup for g4dn.xlarge 

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
source ~/miniconda/etc/profile.d/conda.sh
conda env create -f environment.yml

# 1. Inside the env, purge every conda / pip copy of Torch
conda remove -y pytorch libtorch pytorch-cuda
pip   uninstall -y torch torchaudio torchvision torchtext 2>/dev/null || true
conda install -y fsspec -c conda-forge

# 2. Install the cu124 wheel directly from PyPI
pip install --no-cache-dir --upgrade \
    torch==2.6.0+cu124 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124



# For SFT Smoltalk
python main.py --mode train --algorithm sft --batch_size 4 --learning_rate 1e-5 --num_epochs 1 --output_dir ./models --max_length 1280 --dataset smoltalk --train_ratio 0.1 --val_max_samples 1000 --use_wandb    
#!/usr/bin/env python
"""
Script to generate samples from a checkpoint model using the SmolTalk dataset
"""

import argparse
import os
import torch
import wandb
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import get_dataloaders, clear_disk_cache
from sft import SFTTrainer

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Generate samples from a checkpoint model")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to the checkpoint model")
    parser.add_argument("--dataset", type=str, default="smoltalk",
                        choices=["smoltalk", "warmstart"], 
                        help="Dataset to use for generation")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for data loader")
    parser.add_argument("--val_max_samples", type=int, default=1000,
                        help="Maximum number of validation samples to load")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log metrics to wandb")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save samples as text")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(project="smoltalk-generation", name=f"{args.dataset}-samples")
    
    # Get the dataloader for the specified dataset
    logger.info(f"Loading {args.dataset} dataset...")
    dataloaders = get_dataloaders(
        batch_size=args.batch_size,
        specific_dataset=args.dataset,
        val_max_samples=args.val_max_samples,
    )
    
    # Get validation dataloader
    val_dataloader = dataloaders.get(f"{args.dataset}_val")
    if val_dataloader is None:
        logger.error(f"No validation dataloader found for {args.dataset}")
        return
    
    # Load the model
    logger.info(f"Loading model from {args.checkpoint}")
    model_name = "Qwen/Qwen2.5-0.5B"  # Base model name for tokenizer
    
    # Initialize trainer with the checkpoint model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SFTTrainer(model_name=model_name, device=device)
    success = trainer.load_checkpoint(args.checkpoint)
    
    if not success:
        logger.error("Failed to load checkpoint. Exiting.")
        return
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    samples = trainer.generate_samples(
        val_dataloader,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    
    # Print and log the samples
    for i, sample in enumerate(samples):
        logger.info(f"Sample {i+1}:")
        logger.info(f"Prompt: {sample['prompt']}")
        logger.info(f"Ground truth: {sample['ground_truth']}")
        logger.info(f"Generation: {sample['generation']}")
        logger.info("-" * 40)
        
        # Log to wandb
        if args.use_wandb:
            wandb.log({
                f"sample_{i+1}/prompt": sample['prompt'],
                f"sample_{i+1}/ground_truth": sample['ground_truth'],
                f"sample_{i+1}/generation": sample['generation'],
            })
    
    # Save samples to file if requested
    if args.output_file:
        with open(args.output_file, "w") as f:
            for i, sample in enumerate(samples):
                f.write(f"Sample {i+1}:\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Ground truth: {sample['ground_truth']}\n")
                f.write(f"Generation: {sample['generation']}\n")
                f.write("-" * 40 + "\n\n")
        
        logger.info(f"Samples saved to {args.output_file}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 
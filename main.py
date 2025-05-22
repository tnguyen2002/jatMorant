"""
Main script for running training and evaluation of language models
"""

import argparse
import logging
import os

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate language models")
    
    # Mode arguments
    parser.add_argument("--mode", type=str, required=True, choices=["train", "eval", "all"], 
                        help="Mode to run: train, eval, or all")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, choices=["sft", "dpo", "rloo", "all"], 
                        default="all", help="Algorithm to use")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--max_iterations", type=int, default=500, help="Maximum iterations for RLOO")
    parser.add_argument("--output_dir", type=str, default="./models", help="Output directory")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    
    # Data loading arguments
    parser.add_argument("--max_samples", type=int, default=None, 
                        help="Maximum number of samples to use per dataset")
    parser.add_argument("--force_refresh", action="store_true",
                        help="Force reprocessing of datasets even if cached versions exist")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Maximum sequence length for all examples")
    parser.add_argument("--dataset", type=str, default=None,
                        choices=["smoltalk", "warmstart", "ultrafeedback", "countdown_prompts"],
                        help="Specific dataset to train on (will ignore others)")
    parser.add_argument("--init_from_checkpoint", type=str, default=None,
                        help="Initialize model from a checkpoint (provide path)")
    parser.add_argument("--train_ratio", type=float, default=1.0,
                        help="Ratio of the full dataset to use for training (0.1 = 10%)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Ratio of the training data to use for validation")
    
    # Evaluation arguments
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples for evaluation")
    parser.add_argument("--reward_model_api_key", type=str, default=None, 
                        help="Nemotron reward model API key")
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-0.5B-instruct", 
                        help="Reference model for evaluation")
    
    args = parser.parse_args()
    
    # Load datasets
    from data import get_dataloaders
    dataloaders = get_dataloaders(
        batch_size=args.batch_size,
        task_mode=args.algorithm if args.algorithm != "all" else None,
        force_refresh=args.force_refresh,
        max_samples=args.max_samples,
        max_length=args.max_length,
        specific_dataset=args.dataset,
        train_ratio=args.train_ratio,
        val_split=args.val_split
    )
    
    # Train models if requested
    if args.mode in ["train", "all"]:
        # Train SFT
        if args.algorithm in ["sft", "all"]:
            logger.info("Training SFT model...")
            from sft import SFTTrainer
            
            # Create output directory for SFT
            sft_output_dir = os.path.join(args.output_dir, "sft")
            
            # Train SFT with the SmolTalk dataset
            if ("smoltalk_train" in dataloaders and "smoltalk_val" in dataloaders) and (args.dataset is None or args.dataset == "smoltalk"):
                logger.info("Training SFT model with SmolTalk dataset...")
                sft_trainer = SFTTrainer(
                    model_name="Qwen/Qwen2.5-0.5B",
                    learning_rate=args.learning_rate,
                    num_train_epochs=args.num_epochs,
                    output_dir=sft_output_dir,
                    use_wandb=args.use_wandb,
                    eval_every=100,  # Evaluate every 100 steps
                )
                # Pass validation dataloader to the train method
                sft_trainer.train(
                    train_dataloader=dataloaders["smoltalk_train"],
                    eval_dataloader=dataloaders["smoltalk_val"]
                )
            
            # Train SFT with the Warmstart dataset for Countdown
            if "warmstart" in dataloaders and (args.dataset is None or args.dataset == "warmstart"):
                logger.info("Training SFT model with Warmstart dataset...")
                countdown_sft_output_dir = os.path.join(args.output_dir, "countdown_sft")
                
                # Initialize from checkpoint if provided
                model_name = args.init_from_checkpoint if args.init_from_checkpoint else "Qwen/Qwen2.5-0.5B"
                if args.init_from_checkpoint:
                    logger.info(f"Initializing from checkpoint: {args.init_from_checkpoint}")
                
                countdown_sft_trainer = SFTTrainer(
                    model_name=model_name,
                    learning_rate=args.learning_rate,
                    num_train_epochs=args.num_epochs,
                    output_dir=countdown_sft_output_dir,
                    use_wandb=args.use_wandb,
                )
                countdown_sft_trainer.train(dataloaders["warmstart"])
        
        # Train DPO
        if args.algorithm in ["dpo", "all"]:
            logger.info("Training DPO model...")
            from dpo import DPOTrainer
            
            # Create output directory for DPO
            dpo_output_dir = os.path.join(args.output_dir, "dpo")
            
            # Train DPO with the UltraFeedback dataset
            if "ultrafeedback" in dataloaders and (args.dataset is None or args.dataset == "ultrafeedback"):
                logger.info("Training DPO model with UltraFeedback dataset...")
                dpo_trainer = DPOTrainer(
                    model_name="Qwen/Qwen2.5-0.5B",
                    learning_rate=args.learning_rate / 5,  # Lower learning rate for DPO
                    beta=0.1,
                    num_train_epochs=args.num_epochs,
                    output_dir=dpo_output_dir,
                    use_wandb=args.use_wandb,
                )
                dpo_trainer.train(dataloaders["ultrafeedback"])
        
        # Train RLOO
        if args.algorithm in ["rloo", "all"]:
            logger.info("Training RLOO model...")
            from rloo import RLOOTrainer
            
            # Create output directory for RLOO
            rloo_output_dir = os.path.join(args.output_dir, "rloo")
            
            # Train RLOO with the Countdown prompts dataset
            if "countdown_prompts" in dataloaders and (args.dataset is None or args.dataset == "countdown_prompts"):
                logger.info("Training RLOO model with Countdown prompts dataset...")
                rloo_trainer = RLOOTrainer(
                    model_name="Qwen/Qwen2.5-0.5B",
                    learning_rate=args.learning_rate / 10,  # Lower learning rate for RL
                    num_rollouts=64,
                    max_iterations=args.max_iterations,
                    output_dir=rloo_output_dir,
                    use_wandb=args.use_wandb,
                )
                rloo_trainer.train(dataloaders["countdown_prompts"].dataset)
    
    # Evaluate models if requested
    if args.mode in ["eval", "all"]:
        from evaluate import ModelEvaluator
        
        # Evaluate SFT
        if args.algorithm in ["sft", "all"]:
            # Create output directories for results
            sft_results_dir = os.path.join("results", "sft")
            countdown_sft_results_dir = os.path.join("results", "countdown_sft")
            
            # Evaluate SFT on UltraFeedback
            sft_model_path = os.path.join(args.output_dir, "sft")
            if os.path.exists(sft_model_path):
                logger.info("Evaluating SFT model on UltraFeedback...")
                
                evaluator = ModelEvaluator(
                    model_path=sft_model_path,
                    ref_model_name=args.ref_model,
                    reward_model_api_key=args.reward_model_api_key,
                )
                
                # Run evaluation command
                import subprocess
                subprocess.run([
                    "python", "evaluate.py",
                    "--model_path", sft_model_path,
                    "--ref_model", args.ref_model,
                    "--task", "ultrafeedback",
                    "--num_samples", str(args.num_samples),
                    "--output_path", sft_results_dir,
                    *(["--reward_model_api_key", args.reward_model_api_key] if args.reward_model_api_key else []),
                ])
            
            # Evaluate Countdown SFT
            countdown_sft_model_path = os.path.join(args.output_dir, "countdown_sft")
            if os.path.exists(countdown_sft_model_path):
                logger.info("Evaluating Countdown SFT model...")
                
                # Run evaluation command
                subprocess.run([
                    "python", "evaluate.py",
                    "--model_path", countdown_sft_model_path,
                    "--ref_model", args.ref_model,
                    "--task", "countdown",
                    "--num_samples", str(args.num_samples),
                    "--output_path", countdown_sft_results_dir,
                ])
        
        # Evaluate DPO
        if args.algorithm in ["dpo", "all"]:
            # Create output directory for results
            dpo_results_dir = os.path.join("results", "dpo")
            
            # Evaluate DPO
            dpo_model_path = os.path.join(args.output_dir, "dpo")
            if os.path.exists(dpo_model_path):
                logger.info("Evaluating DPO model...")
                
                # Run evaluation command
                subprocess.run([
                    "python", "evaluate.py",
                    "--model_path", dpo_model_path,
                    "--ref_model", args.ref_model,
                    "--task", "ultrafeedback",
                    "--num_samples", str(args.num_samples),
                    "--output_path", dpo_results_dir,
                    *(["--reward_model_api_key", args.reward_model_api_key] if args.reward_model_api_key else []),
                ])
        
        # Evaluate RLOO
        if args.algorithm in ["rloo", "all"]:
            # Create output directory for results
            rloo_results_dir = os.path.join("results", "rloo")
            
            # Evaluate RLOO
            rloo_model_path = os.path.join(args.output_dir, "rloo")
            if os.path.exists(rloo_model_path):
                logger.info("Evaluating RLOO model...")
                
                # Run evaluation command
                subprocess.run([
                    "python", "evaluate.py",
                    "--model_path", rloo_model_path,
                    "--ref_model", args.ref_model,
                    "--task", "countdown",
                    "--num_samples", str(args.num_samples),
                    "--output_path", rloo_results_dir,
                ])

if __name__ == "__main__":
    main() 
"""
Supervised Fine-Tuning (SFT) Implementation for Qwen 2.5 0.5B model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, get_scheduler
from tqdm.auto import tqdm
import logging
import os
import wandb

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class SFTTrainer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=3,
        warmup_steps=500,
        output_dir="./sft_model",
        use_wandb=False,
        save_every=1000,
        eval_every=500,
        device=None,
    ):
        """
        Initialize the SFT trainer for language model fine-tuning.
        
        Args:
            model_name: Huggingface model name to use (must be Qwen/Qwen2.5-0.5B for this project)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for AdamW optimizer
            max_grad_norm: Maximum gradient norm for gradient clipping
            num_train_epochs: Number of training epochs
            warmup_steps: Number of warmup steps for the learning rate scheduler
            output_dir: Directory to save the model checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            save_every: Save model checkpoint every these many steps
            eval_every: Evaluate model every these many steps
            device: Device to use for training
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.save_every = save_every
        self.eval_every = eval_every
        
        # Track best model
        self.best_eval_loss = float('inf')
        self.best_model_path = None
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        logger.info(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def train(self, train_dataloader, eval_dataloader=None):
        """
        Train the model using SFT.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
        
        Returns:
            Trained model
        """
        # Initialize wandb
        if self.use_wandb:
            wandb.init(project="llm-rl-finetuning", name="sft")
        
        # Calculate total training steps
        total_steps = len(train_dataloader) * self.num_train_epochs
        
        # Initialize learning rate scheduler
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Training loop
        global_step = 0
        for epoch in range(self.num_train_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.num_train_epochs}")
            self.model.train()
            
            # Progress bar
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
            
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,  # Use input_ids as labels for causal LM training
                )
                
                # Get loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Apply loss mask to exclude query tokens from loss calculation
                shift_loss_mask = loss_mask[..., 1:].contiguous()
                
                # Calculate loss only where loss_mask is 1 (i.e., for response tokens)
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                masked_losses = losses.view(shift_labels.size()) * shift_loss_mask
                loss = masked_losses.sum() / (shift_loss_mask.sum() + 1e-8)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "step": global_step,
                    })
                
                # Evaluate
                if eval_dataloader is not None and global_step % self.eval_every == 0:
                    logger.info(f"Evaluating at step {global_step}")
                    eval_loss = self.evaluate(eval_dataloader)
                    logger.info(f"Evaluation loss: {eval_loss}")
                    
                    if self.use_wandb:
                        wandb.log({"eval_loss": eval_loss})
                    
                    # Save only if this is the best model so far
                    if eval_loss < self.best_eval_loss:
                        logger.info(f"New best model with eval loss: {eval_loss:.4f} (previous best: {self.best_eval_loss:.4f})")
                        self.best_eval_loss = eval_loss
                        
                        # Delete previous best model to save space
                        if self.best_model_path and os.path.exists(self.best_model_path):
                            import shutil
                            try:
                                shutil.rmtree(self.best_model_path)
                                logger.info(f"Removed previous best model: {self.best_model_path}")
                            except Exception as e:
                                logger.warning(f"Failed to remove previous checkpoint: {e}")
                        
                        # Save new best model
                        self.best_model_path = f"{self.output_dir}/best-model-{global_step}"
                        self.save_model(self.best_model_path)
                    
                    # Switch back to training mode
                    self.model.train()
            
        # Save final model (only if it's the best we've seen)
        logger.info(f"Training complete. Best model saved at: {self.best_model_path}")
        
        if self.use_wandb:
            wandb.finish()
        
        return self.model
    
    def evaluate(self, eval_dataloader):
        """
        Evaluate the model.
        
        Args:
            eval_dataloader: DataLoader for evaluation
        
        Returns:
            Average evaluation loss
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                loss_mask = batch["loss_mask"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids,
                )
                
                # Get loss
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                # Apply loss mask to exclude query tokens
                shift_loss_mask = loss_mask[..., 1:].contiguous()
                
                # Calculate loss only where loss_mask is 1
                loss_fct = nn.CrossEntropyLoss(reduction="none")
                losses = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                masked_losses = losses.view(shift_labels.size()) * shift_loss_mask
                loss = masked_losses.sum() / (shift_loss_mask.sum() + 1e-8)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(1, num_batches)
    
    def save_model(self, output_path):
        """
        Save model checkpoint.
        
        Args:
            output_path: Path to save the model
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        self.model.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    from data import get_dataloaders
    
    # Get dataloaders
    dataloaders = get_dataloaders(batch_size=8)
    
    # Create trainer
    trainer = SFTTrainer(
        model_name="Qwen/Qwen2.5-0.5B",
        learning_rate=5e-5,
        num_train_epochs=3,
        output_dir="./sft_model",
        use_wandb=False,
    )
    
    # Train model using appropriate dataset
    if "smoltalk" in dataloaders:
        trainer.train(dataloaders["smoltalk"])
    elif "warmstart" in dataloaders:
        trainer.train(dataloaders["warmstart"])
    else:
        print("No suitable dataset found for SFT training") 
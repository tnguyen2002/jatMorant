"""
Direct Preference Optimization (DPO) Implementation for Qwen 2.5 0.5B model
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
from torch.utils.data import DataLoader
import numpy as np

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class DPOTrainer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",
        ref_model_name=None,  # If None, will use a copy of the initial model
        beta=0.1,  # DPO hyperparameter
        learning_rate=5e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_train_epochs=3,
        warmup_steps=500,
        output_dir="./dpo_model",
        use_wandb=False,
        save_every=1000,
        eval_every=500,
        device=None,
    ):
        """
        Initialize the DPO trainer for preference optimization.
        
        Args:
            model_name: Huggingface model name to use (must be Qwen/Qwen2.5-0.5B for this project)
            ref_model_name: Reference model name (if None, will use a copy of the initial model)
            beta: DPO temperature parameter
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
        self.ref_model_name = ref_model_name if ref_model_name is not None else model_name
        self.beta = beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_train_epochs = num_train_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self.save_every = save_every
        self.eval_every = eval_every
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        logger.info(f"Loading policy model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize reference model
        logger.info(f"Loading reference model {self.ref_model_name}...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(self.ref_model_name)
        self.ref_model.to(self.device)
        
        # Make sure reference model is frozen
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def get_logps(self, model, input_ids, attention_mask):
        """
        Compute the log probabilities of a sequence.
        
        Args:
            model: The model to use
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Log probabilities of the sequence
        """
        # with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        
        # Shift logits and input_ids for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_input_ids = input_ids[..., 1:].contiguous()
        
        # Get log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs of the target tokens
        log_probs_token = log_probs.gather(
            -1, shift_input_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Create a mask that ignores padding tokens
        mask = (shift_input_ids != 0).float()
        
        # Compute sequence log probs by summing token log probs
        seq_log_probs = (log_probs_token * mask).sum(-1) / mask.sum(-1)
        
        return seq_log_probs
    
    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps, 
                 reference_chosen_logps, reference_rejected_logps):
        """
        Compute the DPO loss.
        
        Args:
            policy_chosen_logps: Log probs of chosen responses from policy model
            policy_rejected_logps: Log probs of rejected responses from policy model
            reference_chosen_logps: Log probs of chosen responses from reference model
            reference_rejected_logps: Log probs of rejected responses from reference model
            
        Returns:
            DPO loss
        """
        # Compute log ratios for chosen and rejected
        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps
        
        # Compute the DPO loss
        losses = -F.logsigmoid(self.beta * (chosen_logratios - rejected_logratios))
        
        # Return mean loss
        return losses.mean()
    
    def train(self, train_dataloader, eval_dataloader=None):
        """
        Train the model using DPO.
        
        Args:
            train_dataloader: DataLoader for training data
            eval_dataloader: Optional DataLoader for evaluation data
        
        Returns:
            Trained model
        """
        # Initialize wandb
        if self.use_wandb:
            wandb.init(project="llm-rl-finetuning", name="dpo")
        
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
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Get log probs from policy model
                policy_chosen_logps = self.get_logps(
                    self.model, chosen_input_ids, chosen_attention_mask
                )
                policy_rejected_logps = self.get_logps(
                    self.model, rejected_input_ids, rejected_attention_mask
                )
                
                # Get log probs from reference model
                with torch.no_grad():
                    reference_chosen_logps = self.get_logps(
                        self.ref_model, chosen_input_ids, chosen_attention_mask
                    )
                    reference_rejected_logps = self.get_logps(
                        self.ref_model, rejected_input_ids, rejected_attention_mask
                    )
                
                # Compute loss
                loss = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
                
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
                
                # Save checkpoint
                if global_step % self.save_every == 0:
                    logger.info(f"Saving checkpoint at step {global_step}")
                    self.save_model(f"{self.output_dir}/checkpoint-{global_step}")
                
                # Evaluate
                if eval_dataloader is not None and global_step % self.eval_every == 0:
                    logger.info(f"Evaluating at step {global_step}")
                    eval_loss = self.evaluate(eval_dataloader)
                    logger.info(f"Evaluation loss: {eval_loss}")
                    
                    if self.use_wandb:
                        wandb.log({"eval_loss": eval_loss})
                    
                    # Switch back to training mode
                    self.model.train()
            
            # Save model at the end of each epoch
            logger.info(f"Saving model at the end of epoch {epoch+1}")
            self.save_model(f"{self.output_dir}/checkpoint-epoch-{epoch+1}")
        
        # Save final model
        logger.info("Saving final model")
        self.save_model(self.output_dir)
        
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
                chosen_input_ids = batch["chosen_input_ids"].to(self.device)
                chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
                rejected_input_ids = batch["rejected_input_ids"].to(self.device)
                rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)
                
                # Get log probs from policy model
                policy_chosen_logps = self.get_logps(
                    self.model, chosen_input_ids, chosen_attention_mask
                )
                policy_rejected_logps = self.get_logps(
                    self.model, rejected_input_ids, rejected_attention_mask
                )
                
                # Get log probs from reference model
                with torch.no_grad():
                    reference_chosen_logps = self.get_logps(
                        self.ref_model, chosen_input_ids, chosen_attention_mask
                    )
                    reference_rejected_logps = self.get_logps(
                        self.ref_model, rejected_input_ids, rejected_attention_mask
                    )
                
                # Compute loss
                loss = self.dpo_loss(
                    policy_chosen_logps,
                    policy_rejected_logps,
                    reference_chosen_logps,
                    reference_rejected_logps,
                )
                
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
    
    # Check if we have ultrafeedback dataset
    if "ultrafeedback" in dataloaders:
        # Create trainer
        trainer = DPOTrainer(
            model_name="Qwen/Qwen2.5-0.5B",
            learning_rate=1e-5,
            beta=0.1,
            num_train_epochs=3,
            output_dir="./dpo_model",
            use_wandb=False,
        )
        
        # Train model
        trainer.train(dataloaders["ultrafeedback"])
    else:
        print("UltraFeedback dataset not found for DPO training") 
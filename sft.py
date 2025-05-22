"""
Supervised Fine-Tuning (SFT) Implementation for Qwen 2.5 0.5B model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
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
        
        # Training progress tracking
        self.current_epoch = 0
        self.global_step = 0
        
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
        self.global_step = 0
        for epoch in range(self.num_train_epochs):
            self.current_epoch = epoch
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
                
                self.global_step += 1
                progress_bar.set_postfix({"loss": loss.item()})
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "epoch": epoch + 1,
                        "step": self.global_step,
                    })
                
                # Evaluate
                if eval_dataloader is not None and self.global_step % self.eval_every == 0:
                    logger.info(f"Evaluating at step {self.global_step}")
                    eval_loss = self.evaluate(eval_dataloader)
                    logger.info(f"Evaluation loss: {eval_loss}")
                    
                    # Generate samples to see progress
                    samples = self.generate_samples(eval_dataloader, num_samples=2)
                    for i, sample in enumerate(samples):
                        logger.info(f"Sample {i+1}:")
                        logger.info(f"Prompt: {sample['prompt']}")
                        logger.info(f"Ground truth: {sample['ground_truth']}")
                        logger.info(f"Generation: {sample['generation']}")
                        logger.info("-" * 40)
                    
                    if self.use_wandb:
                        wandb.log({"eval_loss": eval_loss})
                        # Log sample generations
                        for i, sample in enumerate(samples):
                            wandb.log({
                                f"sample_{i+1}/prompt": sample['prompt'],
                                f"sample_{i+1}/ground_truth": sample['ground_truth'],
                                f"sample_{i+1}/generation": sample['generation'],
                            })
                    
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
                        self.best_model_path = f"{self.output_dir}/best-model-{self.global_step}"
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
        Save model checkpoint with optimizer state for potential resume.
        
        Args:
            output_path: Path to save the model
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save model weights
        self.model.save_pretrained(output_path)
        
        # Save optimizer and training state
        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'learning_rate': self.learning_rate,
            'epoch': getattr(self, 'current_epoch', 0),
            'global_step': getattr(self, 'global_step', 0),
            'best_eval_loss': self.best_eval_loss
        }, os.path.join(output_path, "training_state.pt"))
        
        logger.info(f"Checkpoint saved to {output_path}")
    
    def generate_samples(self, eval_dataloader, num_samples=3):
        """
        Generate sample texts from validation examples to track training progress.
        
        Args:
            eval_dataloader: Validation dataloader
            num_samples: Number of samples to generate
            
        Returns:
            List of dictionaries containing prompt, ground_truth, and generation
        """
        self.model.eval()
        samples = []
        
        # Get a few examples from validation set
        example_batch = next(iter(eval_dataloader))
        
        with torch.no_grad():
            for i in range(min(num_samples, len(example_batch['input_ids']))):
                # Get prompt (user input)
                input_ids = example_batch['input_ids'][i].to(self.device)
                attention_mask = example_batch['attention_mask'][i].to(self.device)
                loss_mask = example_batch['loss_mask'][i].to(self.device)
                
                # Find where the prompt ends and response begins
                prompt_end = torch.nonzero(loss_mask)[0].item() if torch.sum(loss_mask) > 0 else len(loss_mask)
                prompt_input_ids = input_ids[:prompt_end]
                
                # Generate text
                gen_output = self.model.generate(
                    input_ids=prompt_input_ids.unsqueeze(0),
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.model.config.pad_token_id,
                    attention_mask=attention_mask[:prompt_end].unsqueeze(0),
                )
                
                # Decode generated text
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                generated_text = tokenizer.decode(gen_output[0][prompt_end:], skip_special_tokens=True)
                
                # Get ground truth (from original example)
                ground_truth = tokenizer.decode(input_ids[prompt_end:], skip_special_tokens=True)
                prompt_text = tokenizer.decode(prompt_input_ids, skip_special_tokens=True)
                
                # Add to samples
                samples.append({
                    'prompt': prompt_text,
                    'ground_truth': ground_truth,
                    'generation': generated_text
                })
        
        self.model.train()
        return samples

    def load_checkpoint(self, checkpoint_path):
        """
        Load model and training state from checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
        
        Returns:
            Boolean indicating if loading was successful
        """
        try:
            # Load model weights
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
            self.model.to(self.device)
            
            # Load training state
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path, map_location=self.device)
                
                # Load optimizer state
                self.optimizer.load_state_dict(training_state['optimizer'])
                
                # Load training progress
                self.current_epoch = training_state.get('epoch', 0)
                self.global_step = training_state.get('global_step', 0)
                self.best_eval_loss = training_state.get('best_eval_loss', float('inf'))
                
                logger.info(f"Resumed training from checkpoint at epoch {self.current_epoch+1}, step {self.global_step}")
                logger.info(f"Best eval loss so far: {self.best_eval_loss:.4f}")
                return True
            else:
                logger.warning(f"No training_state.pt found in {checkpoint_path}, loaded only model weights")
                return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False

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
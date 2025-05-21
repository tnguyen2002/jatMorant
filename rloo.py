"""
Reinforcement Learning from Open-ended Outcomes (RLOO) Implementation 
using rule-based reward function for Qwen 2.5 0.5B model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm.auto import tqdm
import logging
import os
import wandb
import time
import numpy as np
from utils.reward_score.countdown import calculate_score

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class RLOOTrainer:
    def __init__(
        self,
        model_name="Qwen/Qwen2.5-0.5B",
        learning_rate=5e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_rollouts=128,
        ent_coef=0.01,  # Entropy coefficient
        value_loss_coef=0.5,  # Value loss coefficient
        ppo_epochs=4,  # Number of PPO epochs
        batch_size=32,
        max_iterations=1000,
        warmup_steps=100,
        output_dir="./rloo_model",
        use_wandb=False,
        save_every=50,
        eval_every=25,
        device=None,
    ):
        """
        Initialize the RLOO trainer for online reinforcement learning.
        
        Args:
            model_name: Huggingface model name to use (must be Qwen/Qwen2.5-0.5B for this project)
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for AdamW optimizer
            max_grad_norm: Maximum gradient norm for gradient clipping
            num_rollouts: Number of rollouts per batch
            ent_coef: Entropy coefficient for exploration
            value_loss_coef: Value loss coefficient
            ppo_epochs: Number of PPO epochs per batch
            batch_size: Batch size for PPO updates
            max_iterations: Maximum number of iterations
            warmup_steps: Number of warmup steps for the learning rate scheduler
            output_dir: Directory to save the model checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            save_every: Save model checkpoint every these many iterations
            eval_every: Evaluate model every these many iterations
            device: Device to use for training
        """
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_rollouts = num_rollouts
        self.ent_coef = ent_coef
        self.value_loss_coef = value_loss_coef
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.max_iterations = max_iterations
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
        logger.info(f"Loading model {model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Value head for critic
        self.value_head = nn.Linear(self.model.config.hidden_size, 1)
        self.value_head.to(self.device)
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_response(self, prompt, max_length=512, temperature=1.0, top_p=0.9):
        """
        Generate a response for a given prompt.
        
        Args:
            prompt: The prompt to generate a response for
            max_length: Maximum length of the response
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated response and the corresponding token IDs
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate response
        outputs = self.model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )
        
        # Get generated response
        generated_ids = outputs.sequences[0]
        response_ids = generated_ids[prompt_length:]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return response, response_ids
    
    def calculate_rewards(self, prompts, responses):
        """
        Calculate rewards for a batch of prompt-response pairs using rule-based reward.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of rewards
        """
        rewards = []
        for prompt, response in zip(prompts, responses):
            # Calculate reward using rule-based reward function
            reward = calculate_score(prompt, response)
            rewards.append(reward)
        
        return rewards
    
    def compute_advantages(self, rewards, values, gamma=0.99, lam=0.95):
        """
        Compute generalized advantage estimation (GAE).
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            gamma: Discount factor
            lam: GAE lambda parameter
            
        Returns:
            Tensor of advantages
        """
        # Convert lists to tensors
        rewards = torch.tensor(rewards, device=self.device)
        values = torch.tensor(values, device=self.device)
        
        # Initialize advantages
        advantages = torch.zeros_like(rewards, device=self.device)
        
        # Compute advantages
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae
        
        # Compute returns
        returns = advantages + values
        
        return advantages, returns
    
    def get_logprobs(self, inputs, targets):
        """
        Compute log probabilities of targets given inputs.
        
        Args:
            inputs: Input token IDs
            targets: Target token IDs
            
        Returns:
            Log probabilities of targets
        """
        with torch.no_grad():
            outputs = self.model(inputs)
            logits = outputs.logits[:, :-1, :]  # Exclude last token prediction
            
            shifted_targets = targets[:, 1:]  # Exclude first token (BOS)
            
            # Get log probabilities
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs_flat = log_probs.reshape(-1, log_probs.size(-1))
            
            # Get log probs of target tokens
            target_log_probs = log_probs_flat[
                torch.arange(log_probs_flat.size(0)),
                shifted_targets.reshape(-1)
            ]
            
            # Reshape to match original shape
            target_log_probs = target_log_probs.reshape(shifted_targets.shape)
            
            # Sum log probs over tokens
            log_probs_sum = target_log_probs.sum(dim=1)
            
            return log_probs_sum
    
    def ppo_update(self, rollout_data):
        """
        Update the model using PPO.
        
        Args:
            rollout_data: Dictionary containing rollout data
            
        Returns:
            Dictionary of update metrics
        """
        # Unpack rollout data
        states = rollout_data["states"]  # Input token IDs
        actions = rollout_data["actions"]  # Response token IDs
        old_logprobs = rollout_data["logprobs"]
        advantages = rollout_data["advantages"]
        returns = rollout_data["returns"]
        
        # Prepare for PPO epochs
        indices = np.arange(len(states))
        num_samples = len(states)
        batch_size = min(self.batch_size, num_samples)
        num_batches = int(np.ceil(num_samples / batch_size))
        
        # Initialize metrics
        metrics = {
            "policy_loss": 0,
            "value_loss": 0,
            "entropy": 0,
            "approx_kl": 0,
            "clipfrac": 0,
        }
        
        # PPO epochs
        for _ in range(self.ppo_epochs):
            # Shuffle indices
            np.random.shuffle(indices)
            
            # Process batches
            for batch_idx in range(num_batches):
                # Get batch indices
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = indices[batch_start:batch_end]
                
                # Get batch data
                batch_states = [states[i] for i in batch_indices]
                batch_actions = [actions[i] for i in batch_indices]
                batch_old_logprobs = old_logprobs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Pad batch states and actions
                batch_states_padded = torch.nn.utils.rnn.pad_sequence(
                    batch_states, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)
                batch_actions_padded = torch.nn.utils.rnn.pad_sequence(
                    batch_actions, batch_first=True, padding_value=self.tokenizer.pad_token_id
                ).to(self.device)
                
                # Forward pass through model
                outputs = self.model(batch_states_padded, return_dict=True)
                
                # Get new log probs
                logits = outputs.logits[:, :-1, :]  # Exclude last token prediction
                shifted_actions = batch_actions_padded[:, 1:]  # Exclude first token (BOS)
                
                # Create a mask for padding tokens
                mask = (shifted_actions != self.tokenizer.pad_token_id).float()
                
                # Get log probabilities
                log_probs = F.log_softmax(logits, dim=-1)
                new_logprobs_token = torch.gather(
                    log_probs, 
                    dim=2, 
                    index=shifted_actions.unsqueeze(-1)
                ).squeeze(-1)
                
                # Apply mask to exclude padding tokens
                new_logprobs_token = new_logprobs_token * mask
                
                # Sum log probs over tokens (excluding padding)
                batch_new_logprobs = (new_logprobs_token.sum(dim=1) / mask.sum(dim=1))
                
                # Get entropy
                entropy = -(log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
                
                # Get last hidden state for value function
                last_hidden_state = outputs.hidden_states[-1]
                
                # Extract value for each sample (use mean pooling for simplicity)
                mean_last_hidden = last_hidden_state.mean(dim=1)
                values = self.value_head(mean_last_hidden).squeeze(-1)
                
                # Compute ratio between old and new policies
                ratio = torch.exp(batch_new_logprobs - batch_old_logprobs)
                
                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * batch_advantages
                
                # Compute policy loss
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss = F.mse_loss(values, batch_returns)
                
                # Compute KL divergence approximation
                approx_kl = ((ratio - 1) - (batch_new_logprobs - batch_old_logprobs)).mean().item()
                
                # Compute clipping fraction
                clipfrac = ((ratio - 1.0).abs() > 0.2).float().mean().item()
                
                # Compute total loss
                loss = policy_loss + self.value_loss_coef * value_loss - self.ent_coef * entropy
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Update metrics
                metrics["policy_loss"] += policy_loss.item() / (self.ppo_epochs * num_batches)
                metrics["value_loss"] += value_loss.item() / (self.ppo_epochs * num_batches)
                metrics["entropy"] += entropy.item() / (self.ppo_epochs * num_batches)
                metrics["approx_kl"] += approx_kl / (self.ppo_epochs * num_batches)
                metrics["clipfrac"] += clipfrac / (self.ppo_epochs * num_batches)
        
        return metrics
    
    def train(self, prompt_dataset):
        """
        Train the model using RLOO.
        
        Args:
            prompt_dataset: Dataset of prompts for online sampling
            
        Returns:
            Trained model
        """
        # Initialize wandb
        if self.use_wandb:
            wandb.init(project="llm-rl-finetuning", name="rloo")
        
        # Initialize learning rate scheduler
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_iterations,
        )
        
        # Training loop
        for iteration in range(self.max_iterations):
            logger.info(f"Starting iteration {iteration+1}/{self.max_iterations}")
            
            # Sample prompts
            prompt_indices = np.random.choice(len(prompt_dataset), self.num_rollouts)
            prompts = [prompt_dataset[i]["prompt"] for i in prompt_indices]
            
            # Collect rollouts
            rollout_data = {
                "states": [],
                "actions": [],
                "rewards": [],
                "logprobs": [],
                "values": [],
            }
            
            # Progress bar for rollouts
            progress_bar = tqdm(prompts, desc="Collecting rollouts")
            
            for prompt in progress_bar:
                # Generate response
                response, response_ids = self.generate_response(
                    prompt, temperature=1.0, top_p=0.9
                )
                
                # Encode prompt for later use
                prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                prompt_ids = prompt_inputs.input_ids[0]
                
                # Combine prompt and response for state representation
                state_ids = torch.cat([prompt_ids, response_ids])
                
                # Calculate reward
                reward = calculate_score(prompt, response)
                
                # Calculate log probability of response
                prompt_tensor = prompt_ids.unsqueeze(0)
                response_tensor = response_ids.unsqueeze(0)
                
                with torch.no_grad():
                    logprob = self.get_logprobs(prompt_tensor, response_tensor).item()
                
                # Estimate value
                with torch.no_grad():
                    outputs = self.model(state_ids.unsqueeze(0), return_dict=True)
                    last_hidden_state = outputs.hidden_states[-1]
                    mean_last_hidden = last_hidden_state.mean(dim=1)
                    value = self.value_head(mean_last_hidden).squeeze().item()
                
                # Add to rollout data
                rollout_data["states"].append(prompt_ids)
                rollout_data["actions"].append(response_ids)
                rollout_data["rewards"].append(reward)
                rollout_data["logprobs"].append(logprob)
                rollout_data["values"].append(value)
                
                progress_bar.set_postfix({"reward": reward})
            
            # Convert lists to tensors
            rollout_data["logprobs"] = torch.tensor(rollout_data["logprobs"], device=self.device)
            
            # Compute advantages and returns
            advantages, returns = self.compute_advantages(
                rollout_data["rewards"], rollout_data["values"]
            )
            rollout_data["advantages"] = advantages
            rollout_data["returns"] = returns
            
            # Update model with PPO
            logger.info("Updating model with PPO...")
            metrics = self.ppo_update(rollout_data)
            
            # Step learning rate scheduler
            lr_scheduler.step()
            
            # Log metrics
            mean_reward = np.mean(rollout_data["rewards"])
            logger.info(f"Mean reward: {mean_reward:.4f}")
            logger.info(f"Policy loss: {metrics['policy_loss']:.4f}")
            logger.info(f"Value loss: {metrics['value_loss']:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    "mean_reward": mean_reward,
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy": metrics["entropy"],
                    "approx_kl": metrics["approx_kl"],
                    "clipfrac": metrics["clipfrac"],
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "iteration": iteration + 1,
                })
            
            # Save checkpoint
            if (iteration + 1) % self.save_every == 0:
                logger.info(f"Saving checkpoint at iteration {iteration+1}")
                self.save_model(f"{self.output_dir}/checkpoint-{iteration+1}")
            
            # Evaluate
            if (iteration + 1) % self.eval_every == 0:
                logger.info(f"Evaluating at iteration {iteration+1}")
                eval_metrics = self.evaluate(prompt_dataset)
                logger.info(f"Evaluation metrics: {eval_metrics}")
                
                if self.use_wandb:
                    wandb.log({
                        "eval_mean_reward": eval_metrics["mean_reward"],
                        "iteration": iteration + 1,
                    })
        
        # Save final model
        logger.info("Saving final model")
        self.save_model(self.output_dir)
        
        if self.use_wandb:
            wandb.finish()
        
        return self.model
    
    def evaluate(self, prompt_dataset, num_samples=50):
        """
        Evaluate the model.
        
        Args:
            prompt_dataset: Dataset of prompts
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Sample prompts
        indices = np.random.choice(len(prompt_dataset), num_samples)
        prompts = [prompt_dataset[i]["prompt"] for i in indices]
        
        rewards = []
        generation_times = []
        
        for prompt in tqdm(prompts, desc="Evaluating"):
            # Measure generation time
            start_time = time.time()
            
            # Generate response (with lower temperature for evaluation)
            response, _ = self.generate_response(
                prompt, temperature=0.7, top_p=0.9
            )
            
            # Calculate generation time
            generation_time = time.time() - start_time
            generation_times.append(generation_time)
            
            # Calculate reward
            reward = calculate_score(prompt, response)
            rewards.append(reward)
        
        # Calculate metrics
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        mean_generation_time = np.mean(generation_times)
        
        metrics = {
            "mean_reward": mean_reward,
            "median_reward": median_reward,
            "mean_generation_time": mean_generation_time,
        }
        
        return metrics
    
    def save_model(self, output_path):
        """
        Save model checkpoint.
        
        Args:
            output_path: Path to save the model
        """
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save model and value head
        self.model.save_pretrained(output_path)
        torch.save(self.value_head.state_dict(), os.path.join(output_path, "value_head.pt"))
        logger.info(f"Model saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    from data import get_dataloaders
    
    # Get dataloaders
    dataloaders = get_dataloaders(batch_size=8)
    
    # Check if we have countdown prompts dataset
    if "countdown_prompts" in dataloaders:
        # Create trainer
        trainer = RLOOTrainer(
            model_name="Qwen/Qwen2.5-0.5B",
            learning_rate=5e-6,
            num_rollouts=64,
            max_iterations=500,
            output_dir="./rloo_model",
            use_wandb=False,
        )
        
        # Get prompt dataset from the dataloader
        prompt_dataset = trainer.train(dataloaders["countdown_prompts"].dataset)
    else:
        print("Countdown prompts dataset not found for RLOO training") 
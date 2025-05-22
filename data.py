from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import os
import sys
import numpy as np  # Added for percentile calculation
import argparse
import pickle
import time

# Add path to reward function
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# Global flag to determine which datasets to load
# Can be overridden via command line or by setting directly
TASK_MODE = "sft"  # Options: "sft", "dpo", "rloo", "all"

# Create cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Load tokenizer (used for both)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# Helper functions for disk caching
def get_cache_path(task, dataset_name):
    """Get path for cached dataset"""
    task_dir = os.path.join(CACHE_DIR, task)
    os.makedirs(task_dir, exist_ok=True)
    return os.path.join(task_dir, f"{dataset_name}.pt")

def save_dataset_to_disk(dataset, task, dataset_name):
    """Save processed dataset to disk"""
    cache_path = get_cache_path(task, dataset_name)
    try:
        # Save PyTorch dataset
        start_time = time.time()
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"Dataset {dataset_name} saved to {cache_path} in {time.time() - start_time:.2f}s")
        return True
    except Exception as e:
        print(f"Error saving dataset {dataset_name} to disk: {e}")
        # If cache file was created but is incomplete, remove it
        if os.path.exists(cache_path):
            os.remove(cache_path)
        return False

def load_dataset_from_disk(task, dataset_name):
    """Load processed dataset from disk if available"""
    cache_path = get_cache_path(task, dataset_name)
    if os.path.exists(cache_path):
        try:
            start_time = time.time()
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Loaded cached dataset {dataset_name} from {cache_path} in {time.time() - start_time:.2f}s")
            return dataset
        except Exception as e:
            print(f"Error loading dataset {dataset_name} from disk: {e}")
            # If cache file is corrupted, remove it
            os.remove(cache_path)
    return None

def clear_disk_cache(task=None):
    """Clear disk cache for specified task or all tasks"""
    if task:
        task_dir = os.path.join(CACHE_DIR, task)
        if os.path.exists(task_dir):
            for filename in os.listdir(task_dir):
                file_path = os.path.join(task_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Disk cache cleared for task: {task}")
    else:
        # Clear all cache
        if os.path.exists(CACHE_DIR):
            for task_name in os.listdir(CACHE_DIR):
                task_dir = os.path.join(CACHE_DIR, task_name)
                if os.path.isdir(task_dir):
                    for filename in os.listdir(task_dir):
                        file_path = os.path.join(task_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
            print("All disk cache cleared")

# Helper functions
def flatten_messages(messages):
    """Concatenate the 'content' fields of a list of message dicts."""
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
        return "\n".join([msg["content"] for msg in messages])
    return messages  # Return as is if not in expected format

# Function to calculate the 95th percentile token length
def calculate_95th_percentile_length(dataset):
    print("Calculating 95th percentile token length for SmolTalk dataset...")
    lengths = []
    sample_size = min(len(dataset), 1000)  # Sample to speed up calculation
    for idx in range(sample_size):
        example = dataset[idx]
        if "messages" in example and len(example["messages"]) >= 2:
            # Get user prompt
            user_message = example["messages"][0]["content"] if example["messages"][0]["role"] == "user" else ""
            # Get response
            assistant_message = example["messages"][1]["content"] if example["messages"][1]["role"] == "assistant" else ""
            
            # Calculate token lengths
            full_text = user_message + "\n" + assistant_message
            tokens = tokenizer(full_text, add_special_tokens=True)["input_ids"]
            lengths.append(len(tokens))
    
    if not lengths:
        print("Warning: Could not calculate token lengths, defaulting to 512")
        return 512
    
    percentile_95 = int(np.percentile(lengths, 95))
    print(f"95th percentile token length: {percentile_95}")
    return percentile_95

# Tokenization function for preference datasets (e.g., UltraFeedback)
def tokenize_ultrafeedback(example):
    prompt = example["prompt"]
    chosen_response = flatten_messages(example["chosen"])
    rejected_response = flatten_messages(example["rejected"])

    # Tokenize with separate query and response to create attention masks
    chosen_full = prompt + "\n" + chosen_response
    chosen_query_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    chosen_enc = tokenizer(chosen_full, truncation=True, padding="max_length", max_length=512)
    
    # Create attention mask where query tokens are masked out for loss
    chosen_attn_mask = chosen_enc["attention_mask"].copy()
    for i in range(chosen_query_len):
        if i < len(chosen_attn_mask):
            chosen_attn_mask[i] = 0  # Mask out query tokens
    
    # Same for rejected
    rejected_full = prompt + "\n" + rejected_response
    rejected_query_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
    rejected_enc = tokenizer(rejected_full, truncation=True, padding="max_length", max_length=512)
    
    rejected_attn_mask = rejected_enc["attention_mask"].copy()
    for i in range(rejected_query_len):
        if i < len(rejected_attn_mask):
            rejected_attn_mask[i] = 0  # Mask out query tokens

    return {
        "prompt": prompt,
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "chosen_loss_mask": chosen_attn_mask,
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "rejected_loss_mask": rejected_attn_mask
    }

# Tokenization for SmolTalk SFT dataset
# Modified to extract only the first user and assistant message from each conversation
# and explicitly limit to 256 tokens for input and 1024 tokens for output
def tokenize_smoltalk(example, max_seq_length=None):
    # SmolTalk dataset contains a list of messages in the 'messages' field
    # Each message has 'content' and 'role' fields
    # We want to extract only the first user and assistant message
    
    if "messages" in example and isinstance(example["messages"], list) and len(example["messages"]) >= 2:
        # In the SmolTalk dataset, typically the first message is from user (index 0)
        # and the second message is from assistant (index 1)
        user_message = example["messages"][0]["content"] if example["messages"][0]["role"] == "user" else ""
        assistant_message = example["messages"][1]["content"] if example["messages"][1]["role"] == "assistant" else ""
        
        prompt = user_message
        response = assistant_message
    else:
        # Fallback to the original fields if messages field doesn't exist or has unexpected format
        prompt = example.get("prompt", example.get("instruction", ""))
        response = example.get("response", example.get("output", ""))
    
    # Tokenize prompt (limit to 256 tokens)
    prompt_tokens = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=256)["input_ids"]
    prompt_len = len(prompt_tokens)
    
    # Tokenize response (limit to 1024 tokens)
    response_tokens = tokenizer(response, add_special_tokens=False, truncation=True, max_length=1024)["input_ids"]
    
    # Total sequence length
    total_seq_length = prompt_len + len(response_tokens)
    
    # Combine into full sequence
    input_ids = prompt_tokens + response_tokens
    
    # Create attention mask
    attention_mask = [1] * total_seq_length
    
    # Create loss mask (0 for prompt tokens, 1 for response tokens)
    loss_mask = [0] * prompt_len + [1] * len(response_tokens)
    
    return {
        "prompt": prompt,
        "response": response,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask
    }

# Tokenization for verifier-based datasets (e.g., Countdown)
def tokenize_countdown(example):
    # Generate prompt from 'nums' and 'target'
    nums = example["nums"]
    target = example["target"]
    prompt = f"Using the numbers {nums}, where you use each number only once, make the target number {target} using +, -, *, and /. Output a single expression."

    # No known ground truth 'completion', so just use empty or dummy string
    completion = example.get("completion", "")

    # Tokenize prompt separately to get its length
    prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_tokens)

    # Tokenize full input (prompt + completion)
    full_input = prompt + completion
    tokens = tokenizer(full_input, truncation=True, padding="max_length", max_length=512)

    # Create loss mask that ignores prompt
    loss_mask = tokens["attention_mask"].copy()
    for i in range(prompt_len):
        if i < len(loss_mask):
            loss_mask[i] = 0

    return {
        "prompt": prompt,
        "completion": completion,
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "loss_mask": loss_mask,
        "verifier_score": example.get("verifier_score", 0.0)
    }


# Dataset Classes

# For SFT
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset, max_length=1024):
        self.data = tokenized_dataset
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get tensors - ensure Long type for input_ids and attention_mask
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(item["attention_mask"], dtype=torch.long)
        loss_mask = torch.tensor(item["loss_mask"], dtype=torch.long)  
        
        # Truncate if necessary
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            loss_mask = loss_mask[:self.max_length]
        
        # Pad if necessary
        if len(input_ids) < self.max_length:
            # Calculate padding needed
            pad_length = self.max_length - len(input_ids)
            
            # Pad input_ids with padding token
            input_ids = torch.cat([input_ids, 
                                   torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)])
            
            # Pad attention_mask with zeros
            attention_mask = torch.cat([attention_mask, 
                                      torch.zeros(pad_length, dtype=torch.long)])
            
            # Pad loss_mask with zeros
            loss_mask = torch.cat([loss_mask, 
                                 torch.zeros(pad_length, dtype=torch.long)])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask
        }

# For DPO
class PreferenceDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "chosen_input_ids": torch.tensor(item["chosen_input_ids"]),
            "chosen_attention_mask": torch.tensor(item["chosen_attention_mask"]),
            "chosen_loss_mask": torch.tensor(item["chosen_loss_mask"]),
            "rejected_input_ids": torch.tensor(item["rejected_input_ids"]),
            "rejected_attention_mask": torch.tensor(item["rejected_attention_mask"]),
            "rejected_loss_mask": torch.tensor(item["rejected_loss_mask"])
        }

# For RLOO
class CountdownPromptsDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "prompt": item["prompt"],
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"])
        }

# Dataset loaders
def load_sft_datasets(force_refresh=False, max_samples=None, max_length=1024, train_ratio=1.0, val_split=0.1, val_max_samples=2000):
    """
    Load datasets needed for SFT
    
    Args:
        force_refresh: If True, reprocess datasets even if cached versions exist
        max_samples: Maximum number of samples to use per dataset (for memory constraints)
        max_length: Maximum sequence length for all examples
        train_ratio: Ratio of the full dataset to use for training (0.1 = 10%)
        val_split: Ratio of the training data to use for validation (not used when using test split)
        val_max_samples: Maximum number of samples to use for validation (limits test set size)
    """
    datasets = {}
    
    # 1. SmolTalk dataset for SFT
    try:
        # Set cache name based on parameters to ensure we don't mix different configurations
        dataset_cache_name = f"smoltalk_ratio{train_ratio}"
        
        # Try to load from disk cache first
        if not force_refresh:
            cached_dataset_train = load_dataset_from_disk("sft", f"{dataset_cache_name}_train")
            cached_dataset_val = load_dataset_from_disk("sft", f"{dataset_cache_name}_test")
            if cached_dataset_train is not None and cached_dataset_val is not None:
                datasets["smoltalk_train"] = cached_dataset_train
                datasets["smoltalk_val"] = cached_dataset_val
        
        # If not in cache or force_refresh, process from scratch
        if "smoltalk_train" not in datasets or "smoltalk_val" not in datasets:
            print("Loading SmolTalk dataset...")
            # Load both train and test splits
            smoltalk_train = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
            smoltalk_test = load_dataset("HuggingFaceTB/smol-smoltalk", split="test")
            
            print(f"SmolTalk dataset loaded with {len(smoltalk_train)} training examples and {len(smoltalk_test)} test examples")
            
            # Apply train ratio to limit dataset size if needed
            if train_ratio < 1.0:
                total_samples = len(smoltalk_train)
                num_samples = int(total_samples * train_ratio)
                print(f"Using {train_ratio*100:.1f}% of SmolTalk training data: {num_samples} examples")
                
                # Override max_samples if smaller than the ratio-limited sample count
                if max_samples is not None:
                    num_samples = min(num_samples, max_samples)
                    print(f"Further limited to {num_samples} examples due to max_samples")
                
                # Sample randomly
                import random
                random.seed(42)  # For reproducibility
                indices = random.sample(range(total_samples), num_samples)
                smoltalk_train = smoltalk_train.select(indices)
            # If not using train_ratio but using max_samples for train
            elif max_samples and len(smoltalk_train) > max_samples:
                print(f"Limiting SmolTalk training dataset to {max_samples} examples")
                smoltalk_train = smoltalk_train.select(range(max_samples))
            
            # Limit test set size if needed (usually much smaller than train)
            if val_max_samples and len(smoltalk_test) > val_max_samples:
                print(f"Limiting SmolTalk test dataset to {val_max_samples} examples for validation")
                # Sample randomly rather than taking first few examples
                random.seed(43)  # Different seed than training
                test_indices = random.sample(range(len(smoltalk_test)), val_max_samples)
                smoltalk_test = smoltalk_test.select(test_indices)
            
            print("Processing SmolTalk dataset with input=256 tokens, output=1024 tokens")
            
            # Process the train dataset
            smoltalk_train_tokenized = smoltalk_train.map(tokenize_smoltalk, batched=False)
            datasets["smoltalk_train"] = SFTDataset(smoltalk_train_tokenized, max_length=max_length)
            print(f"SmolTalk train dataset processed with {len(datasets['smoltalk_train'])} examples")
            
            # Process the test dataset for validation
            smoltalk_test_tokenized = smoltalk_test.map(tokenize_smoltalk, batched=False)
            datasets["smoltalk_val"] = SFTDataset(smoltalk_test_tokenized, max_length=max_length)
            print(f"SmolTalk test dataset (for validation) processed with {len(datasets['smoltalk_val'])} examples")
            
            # Save to disk cache
            save_dataset_to_disk(datasets["smoltalk_train"], "sft", f"{dataset_cache_name}_train")
            save_dataset_to_disk(datasets["smoltalk_val"], "sft", f"{dataset_cache_name}_test")
    except Exception as e:
        print(f"Error loading SmolTalk dataset: {e}")
    
    # 2. WarmStart dataset for SFT (Countdown)
    try:
        # Try to load from disk cache first
        if not force_refresh:
            cached_dataset = load_dataset_from_disk("sft", "warmstart")
            if cached_dataset is not None:
                datasets["warmstart"] = cached_dataset
        
        # If not in cache or force_refresh, process from scratch
        if "warmstart" not in datasets:
            print("Loading WarmStart dataset...")
            warmstart = load_dataset("Asap7772/cog_behav_all_strategies", split="train")
            
            # Apply max_samples limit if specified
            if max_samples and len(warmstart) > max_samples:
                print(f"Limiting WarmStart dataset to {max_samples} examples")
                warmstart = warmstart.select(range(max_samples))
            
            # Let's print the column names to debug
            print(f"WarmStart dataset columns: {warmstart.column_names}")
            
            # Map dataset with the updated tokenize_countdown function
            warmstart_tokenized = warmstart.map(tokenize_countdown, batched=False)
            datasets["warmstart"] = SFTDataset(warmstart_tokenized, max_length=max_length)
            print(f"WarmStart dataset loaded with {len(datasets['warmstart'])} examples")
            print(f"All examples padded/truncated to max_length={max_length}")
            
            # Save to disk cache
            save_dataset_to_disk(datasets["warmstart"], "sft", "warmstart")
    except Exception as e:
        print(f"Error loading WarmStart dataset: {e}")
    
    return datasets

def load_dpo_datasets(force_refresh=False, max_samples=None, max_length=1024):
    """Load datasets needed for DPO with disk caching"""
    datasets = {}
    
    # UltraFeedback preference dataset for DPO
    try:
        # Try to load from disk cache first
        if not force_refresh:
            cached_dataset = load_dataset_from_disk("dpo", "ultrafeedback")
            if cached_dataset is not None:
                datasets["ultrafeedback"] = cached_dataset
        
        # If not in cache or force_refresh, process from scratch
        if "ultrafeedback" not in datasets:
            print("Loading UltraFeedback dataset...")
            ultrafeedback = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
            
            # Apply max_samples limit if specified
            if max_samples and len(ultrafeedback) > max_samples:
                print(f"Limiting UltraFeedback dataset to {max_samples} examples")
                ultrafeedback = ultrafeedback.select(range(max_samples))
            
            ultrafeedback_tokenized = ultrafeedback.map(tokenize_ultrafeedback, batched=False)
            datasets["ultrafeedback"] = PreferenceDataset(ultrafeedback_tokenized)
            print(f"UltraFeedback dataset loaded with {len(datasets['ultrafeedback'])} examples")
            
            # Save to disk cache
            save_dataset_to_disk(datasets["ultrafeedback"], "dpo", "ultrafeedback")
    except Exception as e:
        print(f"Error loading UltraFeedback dataset: {e}")
    
    return datasets

def load_rloo_datasets(force_refresh=False, max_samples=None, max_length=1024):
    """Load datasets needed for RLOO with disk caching"""
    datasets = {}

    try:
        # Try to load from disk cache first
        if not force_refresh:
            cached_dataset = load_dataset_from_disk("rloo", "countdown_prompts")
            if cached_dataset is not None:
                datasets["countdown_prompts"] = cached_dataset

        # If not in cache or force_refresh, process from scratch
        if "countdown_prompts" not in datasets:
            print("Loading Countdown prompts dataset...")
            countdown_prompts = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")

            # Apply max_samples limit if specified
            if max_samples and len(countdown_prompts) > max_samples:
                print(f"Limiting Countdown prompts dataset to {max_samples} examples")
                countdown_prompts = countdown_prompts.select(range(max_samples))

            # Tokenize using updated tokenizer (which constructs prompt from nums + target)
            countdown_prompts_tokenized = countdown_prompts.map(tokenize_countdown, batched=False)

            # Wrap in PyTorch Dataset
            datasets["countdown_prompts"] = CountdownPromptsDataset(countdown_prompts_tokenized)
            print(f"Countdown prompts dataset loaded with {len(datasets['countdown_prompts'])} examples")

            # Save to disk cache
            save_dataset_to_disk(datasets["countdown_prompts"], "rloo", "countdown_prompts")

    except Exception as e:
        print(f"Error loading Countdown prompts dataset: {e}")

    return datasets

# Main function to load all datasets based on task mode
def load_datasets(task_mode=None, force_refresh=False, max_samples=None, max_length=1024, train_ratio=1.0, val_split=0.1, val_max_samples=2000):
    """
    Load datasets based on the task mode
    
    Args:
        task_mode: Which datasets to load. Options: "sft", "dpo", "rloo", "all"
                  If None, use the global TASK_MODE.
        force_refresh: If True, reprocess datasets even if cached versions exist
        max_samples: Maximum number of samples to use per dataset (for memory constraints)
        max_length: Maximum sequence length for all examples
        train_ratio: Ratio of the full dataset to use for training (0.1 = 10%)
        val_split: Ratio of the training data to use for validation
        val_max_samples: Maximum number of samples to use for validation
    
    Returns:
        Dictionary of datasets
    """
    global TASK_MODE
    mode = task_mode or TASK_MODE
    print(f"Loading datasets for task mode: {mode}")
    
    all_datasets = {}
    
    if mode in ["sft", "all"]:
        sft_datasets = load_sft_datasets(force_refresh, max_samples, max_length, train_ratio, val_split, val_max_samples)
        all_datasets.update(sft_datasets)
    
    if mode in ["dpo", "all"]:
        dpo_datasets = load_dpo_datasets(force_refresh, max_samples, max_length)
        all_datasets.update(dpo_datasets)
    
    if mode in ["rloo", "all"]:
        rloo_datasets = load_rloo_datasets(force_refresh, max_samples, max_length)
        all_datasets.update(rloo_datasets)
    
    return all_datasets

# Create DataLoaders
def get_dataloaders(batch_size=8, task_mode=None, force_refresh=False, max_samples=None, max_length=1024, specific_dataset=None, train_ratio=1.0, val_split=0.1, val_max_samples=2000):
    """
    Create DataLoader objects for the loaded datasets
    
    Args:
        batch_size: Batch size for the DataLoaders
        task_mode: Which datasets to load. Options: "sft", "dpo", "rloo", "all"
                  If None, use the global TASK_MODE.
        force_refresh: If True, reprocess datasets even if cached versions exist
        max_samples: Maximum number of samples to use per dataset (for memory constraints)
        max_length: Maximum sequence length for all examples
        specific_dataset: If provided, only load this specific dataset
        train_ratio: Ratio of the full dataset to use for training (0.1 = 10%)
        val_split: Ratio of the training data to use for validation
        val_max_samples: Maximum number of samples to use for validation
    
    Returns:
        Dictionary of DataLoader objects
    """
    datasets = load_datasets(task_mode, force_refresh, max_samples, max_length, train_ratio, val_split, val_max_samples)
    dataloaders = {}
    
    for name, dataset in datasets.items():
        # Skip if a specific dataset is requested and this isn't it
        # We need to handle _train and _val suffixes
        if specific_dataset is not None:
            base_name = name.split('_')[0]
            if base_name != specific_dataset:
                print(f"Skipping dataset {name} as only {specific_dataset} was requested")
                continue
            
        dataloaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name.endswith('_train'),  # Only shuffle training data
            pin_memory=True
        )
    
    return dataloaders

# # Placeholder for rule-based reward function integration
# def get_reward_function():
#     try:
#         # Import the rule-based reward function
#         from reward_score.countdown import calculate_score
        
#         def countdown_reward(prompt, completion):
#             """Wrapper around the rule-based reward function"""
#             return calculate_score(prompt, completion)
        
#         return countdown_reward
#     except ImportError:
#         print("Warning: countdown reward function not found. You need to implement or import it.")
        
#         def dummy_reward(prompt, completion):
#             """Dummy reward function"""
#             return 0.0
        
#         return dummy_reward

# Function to create on-policy preference dataset for DPO (to be used after SFT training)
def create_dpo_dataset(sft_model, tokenizer, prompts_dataset, temperature=0.7, top_p=0.9):
    """
    Create an on-policy preference dataset by sampling from the SFT model.
    This should be called after SFT training.
    
    Args:
        sft_model: The fine-tuned SFT model
        tokenizer: The tokenizer
        prompts_dataset: Dataset containing prompts
        temperature: Sampling temperature
        top_p: Top-p for sampling
        
    Returns:
        A preference dataset for DPO
    """
    print("This function needs an SFT model to generate responses. Use after SFT training.")
    # This is a placeholder - actual implementation would:
    # 1. Sample 2 responses from the SFT model for each prompt
    # 2. Calculate reward scores using the rule-based reward function
    # 3. Label the higher-scoring response as preferred
    # 4. Filter out ties
    # 5. Create a preference dataset
    
    # Example code (pseudocode):
    """
    reward_fn = get_reward_function()
    dpo_examples = []
    
    for prompt in prompts_dataset:
        # Generate 2 responses with temperature > 0
        response1 = generate(sft_model, prompt, temperature=temperature, top_p=top_p)
        response2 = generate(sft_model, prompt, temperature=temperature, top_p=top_p)
        
        # Calculate rewards
        reward1 = reward_fn(prompt, response1)
        reward2 = reward_fn(prompt, response2)
        
        # Skip if tie
        if reward1 == reward2:
            continue
            
        # Label based on reward
        if reward1 > reward2:
            chosen = response1
            rejected = response2
        else:
            chosen = response2
            rejected = response1
            
        dpo_examples.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
    
    # Create dataset
    dpo_dataset = Dataset.from_list(dpo_examples)
    return dpo_dataset
    """
    return None

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Data loading script')
    parser.add_argument('--task', type=str, default=TASK_MODE, 
                        choices=['sft', 'dpo', 'rloo', 'all'],
                        help='Task mode to determine which datasets to load')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for DataLoaders')
    parser.add_argument('--force_refresh', action='store_true',
                        help='Force reprocessing of datasets even if cached versions exist')
    parser.add_argument('--clear_cache', action='store_true',
                        help='Clear disk cache for datasets')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to use per dataset (for memory constraints)')
    parser.add_argument('--max_length', type=int, default=1024,
                        help='Maximum sequence length for all examples')
    parser.add_argument('--specific_dataset', type=str, default=None,
                        help='Only load this specific dataset')
    parser.add_argument('--train_ratio', type=float, default=1.0,
                        help='Ratio of the full dataset to use for training (0.1 = 10%)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Ratio of the training data to use for validation')
    parser.add_argument('--val_max_samples', type=int, default=2000,
                        help='Maximum number of validation samples to use (limits test set size)')
    return parser.parse_args()

if __name__ == "__main__":
    # Parse arguments if running as script
    args = parse_args()
    TASK_MODE = args.task
    
    # Clear cache if requested
    if args.clear_cache:
        if args.task != "all":
            clear_disk_cache(args.task)
        else:
            clear_disk_cache()
    
    # Test loading the datasets
    dataloaders = get_dataloaders(
        batch_size=args.batch_size,
        force_refresh=args.force_refresh,
        max_samples=args.max_samples,
        max_length=args.max_length,
        specific_dataset=args.specific_dataset,
        train_ratio=args.train_ratio,
        val_split=args.val_split,
        val_max_samples=args.val_max_samples
    )
    
    # Print sample from each dataset
    for name, loader in dataloaders.items():
        print(f"\nSample from {name} dataset:")
        batch = next(iter(loader))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

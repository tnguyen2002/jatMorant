from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader
import os
import sys

# Add path to reward function
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# Load tokenizer (used for both)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
tokenizer.pad_token = tokenizer.eos_token

# Helper functions
def flatten_messages(messages):
    """Concatenate the 'content' fields of a list of message dicts."""
    if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], dict):
        return "\n".join([msg["content"] for msg in messages])
    return messages  # Return as is if not in expected format

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
def tokenize_smoltalk(example):
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
    
    # Combine prompt and response
    full_text = prompt + "\n" + response
    
    # Get the token length of the prompt to create the loss mask
    prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_tokens)
    
    # Tokenize the full text
    tokens = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    
    # Create a loss mask that masks out the prompt tokens
    loss_mask = tokens["attention_mask"].copy()
    for i in range(prompt_len):
        if i < len(loss_mask):
            loss_mask[i] = 0
    
    return {
        "prompt": prompt,
        "response": response,
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "loss_mask": loss_mask
    }

# Tokenization for verifier-based datasets (e.g., Countdown)
def tokenize_countdown(example):
    prompt = example["prompt"]
    completion = example["completion"] if "completion" in example else ""
    
    # Tokenize prompt separately to get its length
    prompt_tokens = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    prompt_len = len(prompt_tokens)
    
    # Tokenize full input
    full_input = prompt + completion
    tokens = tokenizer(full_input, truncation=True, padding="max_length", max_length=512)
    
    # Create a loss mask that masks out the prompt tokens
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
        "verifier_score": example.get("verifier_score", 0.0)  # if exists
    }

# Create PyTorch Dataset class for easier batch processing
class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenized_dataset):
        self.data = tokenized_dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "attention_mask": torch.tensor(item["attention_mask"]),
            "loss_mask": torch.tensor(item["loss_mask"])
        }

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

# Load datasets
print("Loading datasets...")

# 1. SmolTalk dataset for SFT - extract only first user-assistant pair from each conversation
try:
    print("Loading SmolTalk dataset...")
    smoltalk = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    print(f"Processing SmolTalk dataset with {len(smoltalk)} examples...")
    smoltalk_tokenized = smoltalk.map(tokenize_smoltalk, batched=False)
    smoltalk_dataset = SFTDataset(smoltalk_tokenized)
    print(f"SmolTalk dataset loaded with {len(smoltalk_dataset)} examples, using only first user-assistant pair")
except Exception as e:
    print(f"Error loading SmolTalk dataset: {e}")
    smoltalk_dataset = None

# 2. UltraFeedback preference dataset for DPO
try:
    ultrafeedback = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
    ultrafeedback_tokenized = ultrafeedback.map(tokenize_ultrafeedback, batched=False)
    ultrafeedback_dataset = PreferenceDataset(ultrafeedback_tokenized)
    print(f"UltraFeedback dataset loaded with {len(ultrafeedback_dataset)} examples")
except Exception as e:
    print(f"Error loading UltraFeedback dataset: {e}")
    ultrafeedback_dataset = None

# 3. WarmStart dataset for SFT (Countdown)
try:
    warmstart = load_dataset("Asap7772/cog_behav_all_strategies", split="train")
    warmstart_tokenized = warmstart.map(tokenize_countdown, batched=False)
    warmstart_dataset = SFTDataset(warmstart_tokenized)
    print(f"WarmStart dataset loaded with {len(warmstart_dataset)} examples")
except Exception as e:
    print(f"Error loading WarmStart dataset: {e}")
    warmstart_dataset = None

# 4. Prompts dataset for RLOO
try:
    countdown_prompts = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    countdown_prompts_tokenized = countdown_prompts.map(
        lambda x: {"prompt": x["prompt"], "input_ids": [], "attention_mask": []}, 
        batched=False
    )
    countdown_prompts_tokenized = countdown_prompts_tokenized.map(tokenize_countdown, batched=False)
    countdown_prompts_dataset = CountdownPromptsDataset(countdown_prompts_tokenized)
    print(f"Countdown prompts dataset loaded with {len(countdown_prompts_dataset)} examples")
except Exception as e:
    print(f"Error loading Countdown prompts dataset: {e}")
    countdown_prompts_dataset = None

# Create DataLoaders
def get_dataloaders(batch_size=8):
    dataloaders = {}
    
    if smoltalk_dataset:
        dataloaders["smoltalk"] = DataLoader(
            smoltalk_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
    
    if ultrafeedback_dataset:
        dataloaders["ultrafeedback"] = DataLoader(
            ultrafeedback_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
    
    if warmstart_dataset:
        dataloaders["warmstart"] = DataLoader(
            warmstart_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
    
    if countdown_prompts_dataset:
        dataloaders["countdown_prompts"] = DataLoader(
            countdown_prompts_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
    
    return dataloaders

# Placeholder for rule-based reward function integration
def get_reward_function():
    try:
        # Import the rule-based reward function
        from reward_score.countdown import calculate_score
        
        def countdown_reward(prompt, completion):
            """Wrapper around the rule-based reward function"""
            return calculate_score(prompt, completion)
        
        return countdown_reward
    except ImportError:
        print("Warning: countdown reward function not found. You need to implement or import it.")
        
        def dummy_reward(prompt, completion):
            """Dummy reward function"""
            return 0.0
        
        return dummy_reward

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

if __name__ == "__main__":
    # Test loading the datasets
    dataloaders = get_dataloaders(batch_size=4)
    
    # Print sample from each dataset
    for name, loader in dataloaders.items():
        print(f"\nSample from {name} dataset:")
        batch = next(iter(loader))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")

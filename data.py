from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer

# Load tokenizer (used for both)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


# Tokenization function for preference datasets (e.g., UltraFeedback)
def flatten_messages(messages):
    """Concatenate the 'content' fields of a list of message dicts."""
    return "\n".join([msg["content"] for msg in messages])

def tokenize_ultrafeedback(example):
    prompt = example["prompt"]
    chosen_response = flatten_messages(example["chosen"])
    rejected_response = flatten_messages(example["rejected"])

    chosen_enc = tokenizer(prompt + "\n" + chosen_response, truncation=True, padding="max_length", max_length=512)
    rejected_enc = tokenizer(prompt + "\n" + rejected_response, truncation=True, padding="max_length", max_length=512)

    return {
        "prompt": prompt,
        "chosen_input_ids": chosen_enc["input_ids"],
        "rejected_input_ids": rejected_enc["input_ids"]
    }


# Tokenization for verifier-based datasets (e.g., Countdown)
def tokenize_countdown(example):
    full_input = example["prompt"] + example["completion"]
    tokens = tokenizer(full_input, truncation=True, padding="max_length", max_length=512)
    return {
        "prompt": example["prompt"],
        "completion": example["completion"],
        "input_ids": tokens["input_ids"],
        "verifier_score": example.get("verifier_score", 0.0)  # if exists
    }

# Load UltraFeedback preference dataset
ultrafeedback = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs")
ultrafeedback_tokenized = ultrafeedback.map(tokenize_ultrafeedback, batched=False)

# Show one example from each
print("UltraFeedback example:")
print(ultrafeedback_tokenized[0])

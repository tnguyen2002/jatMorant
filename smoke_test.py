"""
Quick smoke tests for SFT and DPO training loops.
Run: python -m tests.smoke_test
"""
import torch, os, random
from data import get_dataloaders
from sft import SFTTrainer
from dpo import DPOTrainer

torch.manual_seed(0); random.seed(0)

# --- tiny dataloaders (8 examples) ---------------------------
loaders = get_dataloaders(
    batch_size=2,
    max_samples=8,   # only first 8 rows from each dataset
    task_mode="all",
    force_refresh=False,
    max_length=256,  # shorter to fit on 1 GPU
)

assert "smoltalk" in loaders and "ultrafeedback" in loaders, "Datasets missing!"

# ---------- SFT sanity check ---------------------------------
sft_tr = SFTTrainer(
    num_train_epochs=1,
    learning_rate=1e-4,
    output_dir="./tmp_sft",
    save_every=999999,   # no ckpt
    eval_every=999999,
    use_wandb=False,
)

print("\n▶  SFT 1-batch smoke test")
batch = next(iter(loaders["smoltalk"]))
loss0 = sft_tr.model(
    input_ids=batch["input_ids"].to(sft_tr.device),
    attention_mask=batch["attention_mask"].to(sft_tr.device),
    labels=batch["input_ids"].to(sft_tr.device),
).loss.item()
sft_tr.train(loaders["smoltalk"])   # 1 epoch × 4 mini-batches
loss1 = sft_tr.model(
    input_ids=batch["input_ids"].to(sft_tr.device),
    attention_mask=batch["attention_mask"].to(sft_tr.device),
    labels=batch["input_ids"].to(sft_tr.device),
).loss.item()
print(f"loss before: {loss0:.4f} → after: {loss1:.4f}")

# ---------- DPO sanity check ---------------------------------
dpo_tr = DPOTrainer(
    num_train_epochs=1,
    learning_rate=5e-6,
    beta=0.1,
    output_dir="./tmp_dpo",
    save_every=999999,
    eval_every=999999,
    use_wandb=False,
)

print("\n▶  DPO 1-batch smoke test")
dpo_tr.train(loaders["ultrafeedback"])   # 1 epoch × 4 mini-batches
print("DPO smoke test completed without crash.")

#!/usr/bin/env python3
"""
Generate UltraFeedback leaderboard submissions.

Usage
-----
python gen_ultrafeedback.py \
    --model_dir ./models/sft/best_model \
    --input_ultrafeedback ultrafeedback_test.jsonl \
    --output_jsonl ultrafeedback_mySFT.jsonl \
    --batch_size 4 --max_new_tokens 512 \
    --temperature 0.7 --top_p 0.9 \
    --device cuda   # or "cpu"
"""

import argparse, json, sys, os, torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True,
                   help="Path to your fine-tuned SFT checkpoint directory")
    p.add_argument("--input_ultrafeedback", required=True,
                   help="JSON-Lines file with held-out prompts (from Ed post)")
    p.add_argument("--output_jsonl", required=True,
                   help="Where to write the finished submission")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    # Use tokenizer from model_dir if it exists, otherwise fallback to base model
    tok_path = args.model_dir if os.path.isfile(os.path.join(args.model_dir, "tokenizer_config.json")) else BASE_MODEL_NAME

    print(f"Loading tokenizer from {tok_path}")
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token   # Silence padding warning
    tok.padding_side = "left"

    print(f"Loading model from {args.model_dir} on {args.device}…", file=sys.stderr)
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, trust_remote_code=True).to(args.device)
    model.eval()

    # --- read prompts --------------------------------------------------------
    with open(args.input_ultrafeedback, "r") as f:
        raw = [json.loads(l) for l in f]

    prompts = [ex["prompt"] for ex in raw]

    # --- batched generation --------------------------------------------------
    responses = []
    for i in tqdm(range(0, len(prompts), args.batch_size)):
        batch_prompts = prompts[i : i + args.batch_size]
        enc = tok(batch_prompts, return_tensors="pt",
                  padding=True, truncation=True).to(args.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        for j, ids in enumerate(out):
            # slice away the prompt part → keep only newly generated tokens
            gen_ids = ids[ enc.input_ids.shape[1] : ]
            text = tok.decode(gen_ids, skip_special_tokens=True).strip()
            responses.append(text)

    assert len(responses) == len(raw)

    # --- write leaderboard jsonl --------------------------------------------
    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    with open(args.output_jsonl, "w") as f:
        for ex, resp in zip(raw, responses):
            f.write(json.dumps({"prompt": ex["prompt"],
                                "response": resp}, ensure_ascii=False) + "\n")

    print(f"✅  Wrote {len(responses)} responses to {args.output_jsonl}")

if __name__ == "__main__":
    main()

"""
Evaluation script for fine-tuned language models, comparing against reference models
"""

import torch
import argparse
import logging
import os
import numpy as np
import pandas as pd
import json
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.reward_score import calculate_score
import time
import requests

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(
        self,
        model_path,
        ref_model_name="Qwen/Qwen2.5-0.5B-instruct",
        device=None,
        reward_model_api_key=None,
    ):
        """
        Initialize the model evaluator.
        
        Args:
            model_path: Path to trained model
            ref_model_name: Reference model name
            device: Device to use for evaluation
            reward_model_api_key: API key for Nemotron reward model
        """
        self.model_path = model_path
        self.ref_model_name = ref_model_name
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Initialize trained model
        logger.info(f"Loading trained model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize reference model
        logger.info(f"Loading reference model {ref_model_name}...")
        self.ref_model = AutoModelForCausalLM.from_pretrained(ref_model_name)
        self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # Initialize tokenizer
        tokenizer_base = model_path if os.path.exists(model_path) else "Qwen/Qwen2.5-0.5B"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_base)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # API key for reward model
        self.reward_model_api_key = reward_model_api_key
    
    def generate_response(self, model, prompt, max_length=512, temperature=0.7, top_p=0.9):
        """
        Generate a response for a given prompt.
        
        Args:
            model: The model to use
            prompt: The prompt to generate a response for
            max_length: Maximum length of the response
            temperature: Temperature for sampling
            top_p: Top-p for nucleus sampling
            
        Returns:
            Generated response
        """
        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_length = inputs.input_ids.shape[1]
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Get generated response
        generated_text = self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
        
        return generated_text
    
    def get_nemotron_reward(self, prompt, response):
        """
        Get reward score from Nemotron reward model API.
        
        Args:
            prompt: Prompt
            response: Response
            
        Returns:
            Reward score
        """
        if not self.reward_model_api_key:
            logger.warning("No reward model API key provided, skipping Nemotron evaluation")
            return 0.0
        
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=self.reward_model_api_key
            )
            
            completion = client.chat.completions.create(
                model="nvidia/llama-3.1-nemotron-70b-reward",
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ],
            )
            
            # Extract reward score from completion
            return float(str(completion))
        except Exception as e:
            logger.error(f"Error getting Nemotron reward: {e}")
            return 0.0
    
    def evaluate_ultrafeedback(self, prompts, num_samples=50):
        """
        Evaluate model on UltraFeedback using Nemotron reward model.
        
        Args:
            prompts: List of prompts
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(prompts) > num_samples:
            # Sample a subset
            indices = np.random.choice(len(prompts), num_samples, replace=False)
            prompts = [prompts[i] for i in indices]
        
        results = {
            "prompts": [],
            "trained_model_response": [],
            "reference_model_response": [],
            "trained_model_reward": [],
            "reference_model_reward": [],
            "win": [],
        }
        
        for prompt in tqdm(prompts, desc="Evaluating UltraFeedback"):
            # Generate responses
            trained_response = self.generate_response(self.model, prompt, temperature=0.7)
            reference_response = self.generate_response(self.ref_model, prompt, temperature=0.7)
            
            # Get rewards
            trained_reward = self.get_nemotron_reward(prompt, trained_response)
            reference_reward = self.get_nemotron_reward(prompt, reference_response)
            
            # Determine win
            win = 1 if trained_reward > reference_reward else 0
            
            # Add to results
            results["prompts"].append(prompt)
            results["trained_model_response"].append(trained_response)
            results["reference_model_response"].append(reference_response)
            results["trained_model_reward"].append(trained_reward)
            results["reference_model_reward"].append(reference_reward)
            results["win"].append(win)
        
        # Calculate metrics
        win_rate = np.mean(results["win"])
        mean_trained_reward = np.mean(results["trained_model_reward"])
        mean_reference_reward = np.mean(results["reference_model_reward"])
        
        metrics = {
            "win_rate": win_rate,
            "mean_trained_reward": mean_trained_reward,
            "mean_reference_reward": mean_reference_reward,
            "detailed_results": results,
        }
        
        return metrics
    
    def evaluate_countdown(self, prompts, num_samples=50):
        """
        Evaluate model on Countdown using rule-based reward.
        
        Args:
            prompts: List of prompts
            num_samples: Number of samples to evaluate
            
        Returns:
            Dictionary of evaluation metrics
        """
        if len(prompts) > num_samples:
            # Sample a subset
            indices = np.random.choice(len(prompts), num_samples, replace=False)
            prompts = [prompts[i] for i in indices]
        
        results = {
            "prompts": [],
            "trained_model_response": [],
            "reference_model_response": [],
            "trained_model_reward": [],
            "reference_model_reward": [],
            "win": [],
        }
        
        for prompt in tqdm(prompts, desc="Evaluating Countdown"):
            # Generate responses
            trained_response = self.generate_response(self.model, prompt, temperature=0.7)
            reference_response = self.generate_response(self.ref_model, prompt, temperature=0.7)
            
            # Calculate rewards
            trained_reward = calculate_score(prompt, trained_response)
            reference_reward = calculate_score(prompt, reference_response)
            
            # Determine win
            win = 1 if trained_reward > reference_reward else 0
            
            # Add to results
            results["prompts"].append(prompt)
            results["trained_model_response"].append(trained_response)
            results["reference_model_response"].append(reference_response)
            results["trained_model_reward"].append(trained_reward)
            results["reference_model_reward"].append(reference_reward)
            results["win"].append(win)
        
        # Calculate metrics
        win_rate = np.mean(results["win"])
        mean_trained_reward = np.mean(results["trained_model_reward"])
        mean_reference_reward = np.mean(results["reference_model_reward"])
        
        # Calculate format and verification success rates
        trained_format_success = sum(r >= 0.4 for r in results["trained_model_reward"]) / len(prompts)
        reference_format_success = sum(r >= 0.4 for r in results["reference_model_reward"]) / len(prompts)
        
        trained_verification_success = sum(r >= 0.6 for r in results["trained_model_reward"]) / len(prompts)
        reference_verification_success = sum(r >= 0.6 for r in results["reference_model_reward"]) / len(prompts)
        
        metrics = {
            "win_rate": win_rate,
            "mean_trained_reward": mean_trained_reward,
            "mean_reference_reward": mean_reference_reward,
            "trained_format_success": trained_format_success,
            "reference_format_success": reference_format_success,
            "trained_verification_success": trained_verification_success,
            "reference_verification_success": reference_verification_success,
            "detailed_results": results,
        }
        
        return metrics
    
    def save_results(self, results, output_path="results"):
        """
        Save evaluation results.
        
        Args:
            results: Dictionary of evaluation results
            output_path: Path to save results
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Save summary results as JSON
        summary = {k: v for k, v in results.items() if k != "detailed_results"}
        with open(os.path.join(output_path, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results as CSV
        if "detailed_results" in results:
            df = pd.DataFrame(results["detailed_results"])
            df.to_csv(os.path.join(output_path, "detailed_results.csv"), index=False)
        
        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned language models")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--ref_model", type=str, default="Qwen/Qwen2.5-0.5B-instruct", help="Reference model name")
    parser.add_argument("--task", type=str, choices=["ultrafeedback", "countdown", "both"], default="both", help="Evaluation task")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--reward_model_api_key", type=str, default=None, help="Nemotron reward model API key")
    parser.add_argument("--output_path", type=str, default="results", help="Path to save results")
    
    args = parser.parse_args()
    
    # Load data
    from data import get_dataloaders
    dataloaders = get_dataloaders(batch_size=1)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(
        model_path=args.model_path,
        ref_model_name=args.ref_model,
        reward_model_api_key=args.reward_model_api_key,
    )
    
    if args.task == "ultrafeedback" or args.task == "both":
        # Evaluate UltraFeedback
        if "ultrafeedback" in dataloaders:
            ultrafeedback_data = dataloaders["ultrafeedback"].dataset
            ultrafeedback_prompts = [item["prompt"] for item in ultrafeedback_data]
            
            logger.info("Evaluating UltraFeedback...")
            ultrafeedback_results = evaluator.evaluate_ultrafeedback(
                ultrafeedback_prompts, num_samples=args.num_samples
            )
            
            # Save results
            evaluator.save_results(
                ultrafeedback_results, 
                output_path=os.path.join(args.output_path, "ultrafeedback")
            )
            
            # Print summary
            logger.info(f"UltraFeedback Win Rate: {ultrafeedback_results['win_rate']:.4f}")
            logger.info(f"UltraFeedback Mean Trained Reward: {ultrafeedback_results['mean_trained_reward']:.4f}")
            logger.info(f"UltraFeedback Mean Reference Reward: {ultrafeedback_results['mean_reference_reward']:.4f}")
        else:
            logger.warning("UltraFeedback dataset not found")
    
    if args.task == "countdown" or args.task == "both":
        # Evaluate Countdown
        if "countdown_prompts" in dataloaders:
            countdown_data = dataloaders["countdown_prompts"].dataset
            countdown_prompts = [item["prompt"] for item in countdown_data]
            
            logger.info("Evaluating Countdown...")
            countdown_results = evaluator.evaluate_countdown(
                countdown_prompts, num_samples=args.num_samples
            )
            
            # Save results
            evaluator.save_results(
                countdown_results, 
                output_path=os.path.join(args.output_path, "countdown")
            )
            
            # Print summary
            logger.info(f"Countdown Win Rate: {countdown_results['win_rate']:.4f}")
            logger.info(f"Countdown Mean Trained Reward: {countdown_results['mean_trained_reward']:.4f}")
            logger.info(f"Countdown Mean Reference Reward: {countdown_results['mean_reference_reward']:.4f}")
            logger.info(f"Countdown Trained Format Success: {countdown_results['trained_format_success']:.4f}")
            logger.info(f"Countdown Reference Format Success: {countdown_results['reference_format_success']:.4f}")
            logger.info(f"Countdown Trained Verification Success: {countdown_results['trained_verification_success']:.4f}")
            logger.info(f"Countdown Reference Verification Success: {countdown_results['reference_verification_success']:.4f}")
        else:
            logger.warning("Countdown dataset not found")

if __name__ == "__main__":
    main() 
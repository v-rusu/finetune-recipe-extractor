#!/usr/bin/env python3
"""Evaluate Gemma-3 model for recipe extraction using Unsloth."""

import unsloth

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import argparse
import gc
import json
from typing import List, Dict
from datetime import datetime

import regex as re
from datasets import Dataset
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from dotenv import load_dotenv

from recipe_evaluator import RecipeEvaluator
from vibes_evaluator import VibesEvaluator

# Load environment variables
load_dotenv()

def load_jsonl(path: str) -> List[Dict]:
    """Load a JSONL file and return a list of dictionaries."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def convert_to_chatml(example: dict) -> dict:
    """Convert dataset example to ChatML format, removing reasoning traces."""
    ct = example.get("messages")[2]["content"]
    # Remove reasoning traces since Gemma 3 270M is non-reasoning
    s = re.sub(r"<think>(.|\n)*<\/think>\n\n", "", ct)

    return {
        "conversations": [
            {"role": "system", "content": example["messages"][0]["content"]},
            {"role": "user", "content": example["messages"][1]["content"]},
            {"role": "assistant", "content": s},
        ]
    }

CONFIG = {
    "model_name": "gemma-3-r64-max35-lr5e-4",
    "batch_size": 48,
    "vibes_samples_per_batch": 2,  # Number of samples per batch to evaluate with vibes
    "vibes_model": "deepseek/deepseek-v3.2",
}

def main():
    parser = argparse.ArgumentParser(description="Finetune Gemma-3 for recipe extraction")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./finetuning_dataset.jsonl",
        help="Path to the JSONL dataset file",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=CONFIG["model_name"],
        help="Base model to evaluate",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name,
        max_seq_length = 8192,
        load_in_4bit = False,
    )

    # Setup chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")
    tokenizer.padding_side = "left"  # For batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and process dataset
    print(f"Loading dataset from: {args.dataset}")
    raw_data = load_jsonl(args.dataset)
    print(f"Loaded {len(raw_data)} examples")

    dataset = Dataset.from_list(raw_data)
    dataset = dataset.map(convert_to_chatml)

    # Apply chat template formatting
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    dataset_split = dataset.train_test_split(test_size=0.05, seed=42)

    train_dataset = dataset_split['train']
    eval_dataset = dataset_split['test']

    evaluate_on_eval_dataset(model, tokenizer, eval_dataset, batch_size=CONFIG["batch_size"])

def extract_json_response(generated_text):
    """Extract JSON from generated text."""
    # The decoded text may contain the echoed prompt + "model" + actual response
    if "\nmodel\n" in generated_text:
        json_response = generated_text.split("\nmodel\n")[-1].strip()
    elif "<start_of_turn>model" in generated_text:
        json_response = generated_text.split("<start_of_turn>model")[-1].strip()
    else:
        json_response = generated_text
    
    json_response = json_response.replace("<end_of_turn>", "").strip()
    
    # Extract JSON from the response
    json_match = re.search(r'\{.*\}', json_response, re.DOTALL)
    if json_match:
        json_response = json_match.group(0)
    
    return json_response


def evaluate_on_eval_dataset(model, tokenizer, eval_dataset, batch_size=8):
    """Evaluate the model on the eval dataset using RecipeEvaluator and VibesEvaluator."""
    print("\n" + "=" * 70)
    print("EVALUATING MODEL ON EVAL DATASET")
    print(f"Batch size: {batch_size}")
    print(f"Vibes samples per batch: {CONFIG['vibes_samples_per_batch']}")
    print("=" * 70)
    
    # Initialize evaluators
    recipe_evaluator = RecipeEvaluator()
    vibes_evaluator = None
    
    # Try to initialize vibes evaluator (may fail if API key not set)
    try:
        vibes_evaluator = VibesEvaluator(model=CONFIG["vibes_model"])
        print("✓ VibesEvaluator initialized")
    except ValueError as e:
        print(f"⚠️  VibesEvaluator not available: {e}")
        print("   Continuing with structural validation only...")
    
    num_samples = len(eval_dataset)
    
    # Track all evaluation results
    all_results = {
        "metadata": {
            "model_name": CONFIG["model_name"],
            "batch_size": batch_size,
            "vibes_samples_per_batch": CONFIG["vibes_samples_per_batch"],
            "vibes_model": CONFIG["vibes_model"],
            "total_samples": num_samples,
            "timestamp": datetime.now().isoformat(),
        },
        "samples": []
    }
    
    valid_count = 0
    invalid_count = 0
    all_errors = []
    vibes_count = 0
    
    # Process in batches
    for batch_start in range(0, num_samples, batch_size):
        batch_end = min(batch_start + batch_size, num_samples)
        batch_indices = list(range(batch_start, batch_end))
        batch_num = batch_start // batch_size + 1
        
        print(f"\n--- Batch {batch_num} [{batch_start + 1}-{batch_end}/{num_samples}] ---")
        
        # Prepare batch inputs
        batch_texts = []
        batch_blogposts = []
        
        for idx in batch_indices:
            example = eval_dataset[idx]
            conversations = example["conversations"]
            
            # Extract blogpost from user message
            blogpost = conversations[1]["content"]
            batch_blogposts.append(blogpost)
            
            messages = [
                {"role": "system", "content": conversations[0]["content"]},
                {"role": "user", "content": blogpost},
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ).removeprefix("<bos>")
            batch_texts.append(text)
        
        # Tokenize batch with padding
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        ).to("cuda")
        
        # Generate for batch
        outputs = model.generate(
            **inputs,
            max_new_tokens=1500,
            temperature=1,
            top_p=0.95,
            top_k=64,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        # Process each output in batch
        batch_results = []
        for i, idx in enumerate(batch_indices):
            generated_text = tokenizer.decode(outputs[i], skip_special_tokens=True)
            json_response = extract_json_response(generated_text)
            blogpost = batch_blogposts[i]
            
            print(f"\n[{idx + 1}/{num_samples}] Sample {idx}")
            print(f"  Generated JSON:\n{json_response[:200]}..." if len(json_response) > 200 else f"  Generated JSON:\n{json_response}")
            
            # Structural validation
            is_valid, errors, warnings = recipe_evaluator.evaluate(json_response)
            
            sample_result = {
                "sample_id": idx,
                "batch_num": batch_num,
                "blogpost": blogpost,
                "generated_json": json_response,
                "structural_validation": {
                    "is_valid": is_valid,
                    "errors": errors,
                    "warnings": warnings,
                }
            }
            
            if is_valid:
                valid_count += 1
                print(f"  ✓ Valid")
            else:
                invalid_count += 1
                print(f"  ✗ Invalid: {errors}")
                all_errors.extend(errors)
            
            batch_results.append(sample_result)
        
        # Select samples for vibes evaluation (first N samples from the batch)
        if vibes_evaluator is not None:
            vibes_sample_count = min(CONFIG["vibes_samples_per_batch"], len(batch_results))
            vibes_samples = batch_results[:vibes_sample_count]
            
            print(f"\n  🔍 Running vibes evaluation on {vibes_sample_count} samples from batch {batch_num}...")
            
            for sample_result in vibes_samples:
                try:
                    vibes_eval, raw_response = vibes_evaluator.evaluate(
                        sample_result["blogpost"],
                        sample_result["generated_json"]
                    )
                    
                    sample_result["vibes_evaluation"] = {
                        "evaluation": vibes_eval,
                        "raw_response": raw_response,
                    }
                    
                    vibes_count += 1
                    
                    # Print vibes score
                    if "scores" in vibes_eval:
                        overall_score = vibes_eval["scores"].get("overall", "N/A")
                        print(f"    Sample {sample_result['sample_id']}: Vibes score = {overall_score}/10")
                    elif "error" in vibes_eval:
                        print(f"    Sample {sample_result['sample_id']}: Vibes eval error - {vibes_eval['error']}")
                    
                except Exception as e:
                    print(f"    Sample {sample_result['sample_id']}: Vibes eval failed - {str(e)}")
                    sample_result["vibes_evaluation"] = {
                        "error": str(e)
                    }
        
        # Add all batch results to overall results
        all_results["samples"].extend(batch_results)
    
    # Calculate vibes statistics
    vibes_scores = []
    for sample in all_results["samples"]:
        if "vibes_evaluation" in sample and "evaluation" in sample["vibes_evaluation"]:
            vibes_eval = sample["vibes_evaluation"]["evaluation"]
            if "scores" in vibes_eval and "overall" in vibes_eval["scores"]:
                vibes_scores.append(vibes_eval["scores"]["overall"])
    
    # Add summary statistics
    all_results["summary"] = {
        "total_samples": num_samples,
        "valid_count": valid_count,
        "invalid_count": invalid_count,
        "valid_percentage": valid_count / num_samples * 100 if num_samples > 0 else 0,
        "vibes_evaluated_count": vibes_count,
        "vibes_avg_score": sum(vibes_scores) / len(vibes_scores) if vibes_scores else None,
        "vibes_min_score": min(vibes_scores) if vibes_scores else None,
        "vibes_max_score": max(vibes_scores) if vibes_scores else None,
    }
    
    # Save all results to JSON file
    output_filename = f"eval_results_{CONFIG['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Saved all evaluation data to: {output_filename}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total samples: {num_samples}")
    print(f"Valid: {valid_count} ({valid_count/num_samples*100:.1f}%)")
    print(f"Invalid: {invalid_count} ({invalid_count/num_samples*100:.1f}%)")
    
    if vibes_count > 0:
        print(f"\nVibes Evaluation:")
        print(f"  Samples evaluated: {vibes_count}")
        if vibes_scores:
            print(f"  Average score: {all_results['summary']['vibes_avg_score']:.2f}/10")
            print(f"  Score range: {all_results['summary']['vibes_min_score']:.1f} - {all_results['summary']['vibes_max_score']:.1f}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Finetune Gemma-3 model for recipe extraction using Unsloth."""

import unsloth

import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

import argparse
import gc
import json
from typing import List, Dict

import regex as re
import torch
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only

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

# CONFIG = {
#     "lora_rank": 128,
#     "max_steps": 50,
#     "learning_rate": 2e-5,
#     "save_dir": "gemma-3-r128-max50-lr2e-5",
# }

# CONFIG = {
#     "lora_rank": 128,
#     "max_steps": 150,
#     "learning_rate": 2e-5,
#     "save_dir": "gemma-3-r128-max150-lr2e-5",
# }

# CONFIG = {
#     "lora_rank": 128,
#     "max_steps": 150,
#     "learning_rate": 1e-5,
#     "save_dir": "gemma-3-r128-max150-lr1e-5",
# }

# CONFIG = {
#     "lora_rank": 256,
#     "max_steps": 50,
#     "learning_rate": 2e-5,
#     "save_dir": "gemma-3-r256-max50-lr2e-5",
# }

# CONFIG = {
#     "lora_rank": 256,
#     "max_steps": 150,
#     "learning_rate": 2e-5,
#     "save_dir": "gemma-3-r256-max150-lr2e-5",
# }

# CONFIG = {
#     "lora_rank": 256,
#     "max_steps": 150,
#     "learning_rate": 1e-5,
#     "save_dir": "gemma-3-r256-max150-lr1e-5",
# }

# CONFIG = {
#     "lora_rank": 128,
#     "max_steps": 50,
#     "learning_rate": 1e-5,
#     "save_dir": "gemma-3-r128-max50-lr1e-5",
# }

# Second best model so far
# CONFIG = {
#     "lora_rank": 128,
#     "max_steps": 50,
#     "learning_rate": 5e-4,
#     "save_dir": "gemma-3-r128-max50-lr5e-4",
# }

## This is the best model so far
# CONFIG = {
#     "lora_rank": 64,
#     "max_steps": 50,
#     "learning_rate": 5e-4,
#     "save_dir": "gemma-3-r64-max50-lr5e-4",
# }

# Worse than the previous one
# CONFIG = {
#     "lora_rank": 64,
#     "max_steps": 50,
#     "gradient_accumulation_steps": 1,
#     "learning_rate": 5e-4,
#     "save_dir": "gemma-3-r64-max50-ga1-lr5e-4",
# }

# CONFIG = {
#     "lora_rank": 64,
#     "max_steps": 50,
#     "gradient_accumulation_steps": 6,
#     "learning_rate": 5e-4,
#     "save_dir": "gemma-3-r64-max50-ga6-lr5e-4",
# }

CONFIG = {
    "lora_rank": 64, # Try lora 32 tomorrow
    "max_steps": 35,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "save_dir": "gemma-3-r64-max35-lr5e-4",
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
        default="unsloth/gemma-3-270m-it",
        help="Base model to finetune",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=8192,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=CONFIG["lora_rank"],
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Per-device training batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=CONFIG["gradient_accumulation_steps"] if CONFIG["gradient_accumulation_steps"] is not None else 4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=CONFIG["max_steps"],
        help="Maximum training steps (set to -1 for full epochs)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=CONFIG["learning_rate"],
        help="Learning rate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=CONFIG["save_dir"],
        help="Directory to save the final model",
    )
    parser.add_argument(
        "--save-gguf",
        action="store_true",
        help="Save model in GGUF format",
    )
    parser.add_argument(
        "--gguf-quantization",
        type=str,
        default="Q8_0",
        choices=["Q8_0", "BF16", "F16"],
        help="GGUF quantization method",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load model in 8-bit quantization",
    )
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_4bit,
        load_in_8bit=args.load_8bit,
        full_finetuning=False,
    )

    # Add LoRA adapters
    print(f"Adding LoRA adapters with rank {args.lora_rank}")
    model = FastModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_rank,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Setup chat template
    tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

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

    # Setup trainer
    print("Setting up trainer...")
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=sft_config,
    )

    # Train only on assistant responses
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    # Show memory stats
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        start_gpu_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        print(f"{start_gpu_memory} GB of memory reserved.")

    # Train
    print("Starting training...")
    trainer_stats = trainer.train()

    # Show training stats
    if torch.cuda.is_available():
        used_memory = round(
            torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
        )
        used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
        print(f"Peak reserved memory = {used_memory} GB.")
        print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
        print(f"Peak reserved memory % of max memory = {used_percentage} %.")

    # Save model
    print(f"\nSaving model to: {args.save_dir}")
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # Save GGUF if requested
    if args.save_gguf:
        gguf_dir = f"{args.save_dir}-gguf"
        print(f"Saving GGUF to: {gguf_dir}")
        model.save_pretrained_gguf(
            gguf_dir,
            tokenizer,
            quantization_method=args.gguf_quantization,
        )

    print("\nTraining complete!")

if __name__ == "__main__":
    main()

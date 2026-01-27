#!/usr/bin/env python3
"""Finetune Gemma-3 model for recipe extraction using Unsloth with W&B tracking."""

import argparse
import gc
import json
from typing import List, Dict

import regex as re
import torch
import wandb
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


def get_config_value(config, key, default):
    """Safely get config value with default, handling None."""
    val = config.get(key, default)
    return default if val is None else val


def train(config=None):
    """Main training function, compatible with wandb sweeps."""
    # Initialize wandb
    with wandb.init(config=config):
        # Get config from wandb (allows sweep to override)
        config = wandb.config

        # Extract all config values with safe defaults
        model_name = get_config_value(config, "model_name", "unsloth/gemma-3-270m-it")
        dataset_path = get_config_value(config, "dataset", "./finetuning_dataset.jsonl")
        max_seq_length = get_config_value(config, "max_seq_length", 8192)
        lora_rank = get_config_value(config, "lora_rank", 128)
        lora_alpha = get_config_value(config, "lora_alpha", lora_rank)
        lora_dropout = get_config_value(config, "lora_dropout", 0)
        batch_size = get_config_value(config, "batch_size", 4)
        gradient_accumulation_steps = get_config_value(config, "gradient_accumulation_steps", 4)
        max_steps = get_config_value(config, "max_steps", -1)
        num_epochs = get_config_value(config, "num_epochs", 1)
        learning_rate = get_config_value(config, "learning_rate", 2e-5)
        weight_decay = get_config_value(config, "weight_decay", 0.001)
        warmup_steps = get_config_value(config, "warmup_steps", 5)
        lr_scheduler_type = get_config_value(config, "lr_scheduler_type", "linear")
        optimizer = get_config_value(config, "optimizer", "adamw_8bit")
        output_dir = get_config_value(config, "output_dir", "outputs")
        load_4bit = get_config_value(config, "load_4bit", False)
        load_8bit = get_config_value(config, "load_8bit", False)
        use_rslora = get_config_value(config, "use_rslora", False)
        save_dir = get_config_value(config, "save_dir", None)
        save_gguf = get_config_value(config, "save_gguf", False)
        gguf_quantization = get_config_value(config, "gguf_quantization", "Q8_0")

        # Load model
        print(f"Loading model: {model_name}")
        model, tokenizer = FastModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            load_in_4bit=load_4bit,
            load_in_8bit=load_8bit,
            full_finetuning=False,
        )

        # Add LoRA adapters
        print(f"Adding LoRA adapters with rank {lora_rank}")
        model = FastModel.get_peft_model(
            model,
            r=lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=use_rslora,
            loftq_config=None,
        )

        # Setup chat template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

        # Load and process dataset
        print(f"Loading dataset from: {dataset_path}")
        raw_data = load_jsonl(dataset_path)
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

        # Clear memory before training
        gc.collect()
        torch.cuda.empty_cache()

        # Setup trainer
        print("Setting up trainer...")
        sft_config = SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=max_steps if max_steps > 0 else None,
            num_train_epochs=num_epochs if max_steps <= 0 else 1,
            learning_rate=learning_rate,
            logging_steps=1,
            optim=optimizer,
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            seed=3407,
            output_dir=output_dir,
            report_to="wandb",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset,
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

            # Log GPU info to wandb
            wandb.log({
                "gpu_name": gpu_stats.name,
                "gpu_max_memory_gb": max_memory,
                "gpu_reserved_memory_gb": start_gpu_memory,
            })

        # Train
        print("Starting training...")
        trainer_stats = trainer.train()

        # Log final metrics
        if torch.cuda.is_available():
            used_memory = round(
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3
            )
            used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
            used_percentage = round(used_memory / max_memory * 100, 3)

            wandb.log({
                "train_runtime_seconds": trainer_stats.metrics["train_runtime"],
                "peak_memory_gb": used_memory,
                "peak_memory_for_training_gb": used_memory_for_lora,
                "peak_memory_percentage": used_percentage,
                "final_train_loss": trainer_stats.metrics.get("train_loss", None),
            })

            print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
            print(f"Peak reserved memory = {used_memory} GB.")
            print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
            print(f"Peak reserved memory % of max memory = {used_percentage} %.")

        # Save model if configured
        if save_dir:
            print(f"\nSaving model to: {save_dir}")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

            # Save GGUF if requested
            if save_gguf:
                gguf_dir = f"{save_dir}-gguf"
                print(f"Saving GGUF to: {gguf_dir}")
                model.save_pretrained_gguf(
                    gguf_dir,
                    tokenizer,
                    quantization_method=gguf_quantization,
                )

        print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Finetune Gemma-3 for recipe extraction with W&B tracking"
    )
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
        default=128,
        help="LoRA rank (r parameter)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (defaults to lora-rank)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0,
        help="LoRA dropout",
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
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum training steps (set to -1 for full epochs)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (used when max-steps is -1)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="linear",
        choices=["linear", "cosine", "constant", "constant_with_warmup"],
        help="Learning rate scheduler type",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw_8bit",
        choices=["adamw_8bit", "adamw_torch", "sgd", "adafactor"],
        help="Optimizer to use",
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
        default=None,
        help="Directory to save the final model (optional)",
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
    parser.add_argument(
        "--use-rslora",
        action="store_true",
        help="Use RSLoRA for rank-stabilized training",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="recipe-extractor-finetune",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (team or username)",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        default=None,
        help="W&B sweep ID to join (for running sweep agents)",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=None,
        help="Number of sweep runs to execute",
    )

    args = parser.parse_args()

    # Build config dict from args
    config = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha if args.lora_alpha else args.lora_rank,
        "lora_dropout": args.lora_dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_steps": args.max_steps,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "optimizer": args.optimizer,
        "output_dir": args.output_dir,
        "save_dir": args.save_dir,
        "save_gguf": args.save_gguf,
        "gguf_quantization": args.gguf_quantization,
        "load_4bit": args.load_4bit,
        "load_8bit": args.load_8bit,
        "use_rslora": args.use_rslora,
    }

    if args.sweep_id:
        # Join existing sweep as an agent
        wandb.agent(
            args.sweep_id,
            function=train,
            project=args.wandb_project,
            entity=args.wandb_entity,
            count=args.sweep_count,
        )
    else:
        # Single run with provided config
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=config,
        )
        train(config)


if __name__ == "__main__":
    main()

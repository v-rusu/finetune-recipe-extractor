#!/usr/bin/env python3
"""Finetune Gemma-3 model for recipe extraction using Unsloth."""

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
        default=128,
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
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="gemma-3",
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
    parser.add_argument(
        "--skip-inference-test",
        action="store_true",
        help="Skip the inference test after training",
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

    # Clear memory before training
    gc.collect()
    torch.cuda.empty_cache()

    # Setup trainer
    print("Setting up trainer...")
    sft_config = SFTConfig(
        dataset_text_field="text",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=5,
        max_steps=args.max_steps if args.max_steps > 0 else None,
        num_train_epochs=args.num_epochs if args.max_steps <= 0 else 1,
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

    # Run inference test
    if not args.skip_inference_test:
        print("\nRunning inference test...")
        run_inference_test(model, tokenizer)

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


def run_inference_test(model, tokenizer):
    """Run a quick inference test on the trained model."""
    system_prompt = """You are a recipe extraction assistant. Your task is to analyze the provided text and extract recipe information in LD-JSON format following the schema.org Recipe specification.

Given a piece of text that describes a cooking recipe (possibly including a long blog post, personal stories, comments, etc.), extract a single Recipe object in JSON-LD format.

Return ONLY valid JSON in the following format:
{
  "@context": "https://schema.org",
  "@type": "Recipe",
  "name": "Recipe Name",
  "description": "Recipe description",
  "recipeIngredient": ["quantity of ingredient 1", "quantity of ingredient 2", ...],
  "recipeInstructions": [
    {
      "@type": "HowToStep",
      "text": "Step 1 instruction"
    }
  ],
  "prepTime": "PT15M",
  "cookTime": "PT30M",
  "totalTime": "PT45M",
  "recipeYield": "4 servings",
  "recipeCategory": "Main Course",
  "recipeCuisine": "Italian",
  "keywords": "pasta, dinner"
}

Include as many fields as you can extract from the text. If you cannot find recipe information, return an empty object.

Do not include any explanatory text, only the JSON."""

    user_content = """Extract the recipe data from this text:

Simple Pancakes

Ingredients:
- 1 cup flour
- 1 egg
- 1 cup milk
- 2 tbsp butter, melted

Instructions:
1. Mix flour and egg in a bowl.
2. Add milk and melted butter, stir until smooth.
3. Heat a pan over medium heat.
4. Pour batter and cook until bubbles form, then flip.
5. Serve warm with syrup.

Prep time: 5 minutes
Cook time: 15 minutes
Serves: 4
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    ).removeprefix("<bos>")

    from transformers import TextStreamer

    print("Model output:")
    print("-" * 50)
    _ = model.generate(
        **tokenizer(text, return_tensors="pt").to("cuda"),
        max_new_tokens=1000,
        temperature=1,
        top_p=0.95,
        top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )
    print("-" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Create a finetuning dataset from llm_output_reasoning JSON files.
Each JSON file contains both 'recipe' and 'reasoning' fields.
Combines with blog posts to create training examples.
"""

import argparse
import sys
import json
import random
from pathlib import Path


def create_training_example(blog_text, recipe_json, reasoning, system_prompt=None):
    """
    Create training example in messages format.

    Format:
    - User: blog post text
    - Assistant: <think>reasoning</think> + JSON recipe
    """
    if system_prompt is None:
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

    # Wrap reasoning in <think> tags if not already wrapped
    if reasoning and not reasoning.strip().startswith('<think>'):
        reasoning_wrapped = f"<think>\n{reasoning.strip()}\n</think>"
    else:
        reasoning_wrapped = reasoning

    # Combine reasoning and JSON for assistant response
    assistant_response = f"{reasoning_wrapped}\n\n{json.dumps(recipe_json, indent=2, ensure_ascii=False)}"

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Extract recipe information from this text:\n\n{blog_text}"},
            {"role": "assistant", "content": assistant_response}
        ]
    }


def main():
    parser = argparse.ArgumentParser(
        description='Create finetuning dataset from llm_output_reasoning JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create dataset with default settings
  python 04_generate_finetuning_dataset.py

  # Create with 80/20 train/test split
  python 04_generate_finetuning_dataset.py --split 0.8

  # Use custom directories
  python 04_generate_finetuning_dataset.py \
    --blog-dir ./generated_recipes_blog \
    --llm-dir ./llm_output_reasoning \
    --output ./dataset_v2.jsonl

  # Process only first 100 examples
  python 04_generate_finetuning_dataset.py --limit 100
        """
    )

    parser.add_argument('--blog-dir', default='./generated_recipes_blog',
                       help='Directory with blog post text files (default: ./generated_recipes_blog)')
    parser.add_argument('--llm-dir', default='./llm_output_reasoning',
                       help='Directory with LLM output JSON files (default: ./llm_output_reasoning)')
    parser.add_argument('--output', default='./finetuning_dataset.jsonl',
                       help='Output JSONL file (default: ./finetuning_dataset.jsonl)')
    parser.add_argument('--system-prompt', default=None,
                       help='Custom system prompt')
    parser.add_argument('--split', type=float, default=None,
                       help='Create train/test split (e.g., 0.8 for 80%% train, 20%% test)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of examples to process (default: process all)')

    args = parser.parse_args()

    # Setup paths
    blog_dir = Path(args.blog_dir)
    llm_dir = Path(args.llm_dir)
    output_path = Path(args.output)

    # Validate directories exist
    for dir_path, name in [(blog_dir, 'Blog'), (llm_dir, 'LLM output')]:
        if not dir_path.exists():
            print(f"Error: {name} directory '{dir_path}' does not exist", file=sys.stderr)
            sys.exit(1)

    # Find all LLM output JSON files
    llm_files = sorted(list(llm_dir.glob('*.json')))

    if not llm_files:
        print(f"No .json files found in {llm_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(llm_files)} LLM output file(s)")
    print()

    # Process each LLM file and match with blog post
    examples = []
    matched_count = 0
    missing_blog_count = 0
    error_count = 0

    for i, llm_file in enumerate(llm_files, 1):
        if args.limit and i > args.limit:
            print(f"\nReached limit of {args.limit} examples")
            break

        recipe_name = llm_file.stem

        # Find corresponding blog file
        blog_file = blog_dir / f"{recipe_name}.txt"

        # Check if blog file exists
        if not blog_file.exists():
            print(f"[{i}/{len(llm_files)}] ⚠ Missing blog post for: {recipe_name}")
            missing_blog_count += 1
            continue

        try:
            # Read blog post
            with open(blog_file, 'r', encoding='utf-8') as f:
                blog_text = f.read()

            # Read LLM output JSON (contains both recipe and reasoning)
            with open(llm_file, 'r', encoding='utf-8') as f:
                llm_data = json.load(f)

            # Extract recipe and reasoning
            recipe_json = llm_data.get('recipe')
            reasoning = llm_data.get('reasoning', '')

            if not recipe_json:
                print(f"[{i}/{len(llm_files)}] ⚠ No recipe data in: {recipe_name}")
                error_count += 1
                continue

            # Create training example
            example = create_training_example(blog_text, recipe_json, reasoning, args.system_prompt)

            examples.append(example)
            matched_count += 1

            # Progress indicator
            if i % 100 == 0:
                print(f"[{i}/{len(llm_files)}] Processed {matched_count} examples...")

        except Exception as e:
            print(f"[{i}/{len(llm_files)}] ✗ Error processing {recipe_name}: {e}")
            error_count += 1

    print(f"\nMatched and processed {matched_count} complete examples")

    if not examples:
        print("Error: No complete examples found", file=sys.stderr)
        sys.exit(1)

    # Handle train/test split if requested
    if args.split:
        random.shuffle(examples)

        split_index = int(len(examples) * args.split)
        train_examples = examples[:split_index]
        test_examples = examples[split_index:]

        # Determine output paths
        base_path = output_path.with_suffix('')
        train_path = Path(f"{base_path}_train.jsonl")
        test_path = Path(f"{base_path}_test.jsonl")

        # Write train set
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        # Write test set
        with open(test_path, 'w', encoding='utf-8') as f:
            for example in test_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"\n✓ Train set saved: {train_path} ({len(train_examples)} examples)")
        print(f"✓ Test set saved: {test_path} ({len(test_examples)} examples)")

    else:
        # Write all examples to single file
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')

        print(f"\n✓ Dataset saved: {output_path} ({len(examples)} examples)")

    # Summary
    print("\n" + "=" * 80)
    print(f"Dataset creation complete!")
    print(f"  ✓ Complete examples: {matched_count}")
    print(f"  ⚠ Missing blog posts: {missing_blog_count}")
    print(f"  ✗ Errors: {error_count}")

    if args.split:
        print(f"\nSplit: {int(args.split * 100)}% train / {int((1 - args.split) * 100)}% test")


if __name__ == '__main__':
    main()

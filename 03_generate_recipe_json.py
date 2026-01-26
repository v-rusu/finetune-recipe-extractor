#!/usr/bin/env python3
"""
Process scraped text files with LMStudio to extract recipe LD-JSON.
"""

from openai import OpenAI
import argparse
import sys
import json
import re
import time
from pathlib import Path
from json_repair import JSONRepairer


SYSTEM_PROMPT = """You are a recipe extraction assistant. Your task is to analyze the provided text and extract recipe information in LD-JSON format following the schema.org Recipe specification.

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


def format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def process_text_with_llm(client, text, model="google/gemma-3-1b"):
    """Send text to LMStudio and get recipe JSON back."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract recipe information from this text:\n\n{text}"}
            ],
            temperature=0.1,
            max_tokens=4000
        )

        content = response.choices[0].message.content.strip()
        reasoning = response.choices[0].message.reasoning_content.strip()

        # Parse the JSON content
        repairer = JSONRepairer()
        result = repairer.repair(content)

        if not result.success:
            print(f"  ✗ JSON repair failed: {result.error}", file=sys.stderr)
            return None, None

        return result.data, reasoning

    except Exception as e:
        print(f"  ✗ Error processing with LLM: {e}", file=sys.stderr)
        return None, None


def main():
    parser = argparse.ArgumentParser(
        description='Process scraped text files with LMStudio to extract recipe JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process files from default scraped_output directory
  python 03_generate_recipe_json.py

  # Specify custom input and output directories
  python 03_generate_recipe_json.py --input scraped_output --output llm_output

  # Use custom LMStudio API endpoint
  python 03_generate_recipe_json.py --base-url http://localhost:1234/v1

  # Specify model name
  python 03_generate_recipe_json.py --model llama-3.2-3b-instruct
        """
    )

    parser.add_argument('--input', '-i', default='./generated_recipes_blog',
                       help='Input directory with scraped text files (default: ./scraped_output)')
    parser.add_argument('--output', '-o', default='./llm_output_reasoning',
                       help='Output directory for LLM-generated JSON (default: ./llm_output)')
    parser.add_argument('--base-url', default='http://localhost:1234/v1',
                       help='LMStudio API base URL (default: http://localhost:1234/v1)')
    parser.add_argument('--model', default='unsloth/Qwen3-14B-unsloth-bnb-4bit',
                       help='Model name to use (default: unsloth/Qwen3-14B-unsloth-bnb-4bit)')
    parser.add_argument('--api-key', default='lm-studio',
                       help='API key (default: lm-studio)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of recipes to process (default: process all)')
    parser.add_argument('--skip', type=int, default=0,
                       help='Skip first N recipes (default: 0)')

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all text files
    text_files = sorted(list(input_dir.glob('*.txt')))

    if not text_files:
        print(f"No .txt files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    # Initialize OpenAI client for LMStudio
    try:
        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key
        )
        print(f"Connected to LMStudio at {args.base_url}")
        print(f"Using model: {args.model}\n")
    except Exception as e:
        print(f"Error connecting to LMStudio: {e}", file=sys.stderr)
        sys.exit(1)

    # Process each text file
    success_count = 0
    empty_count = 0
    error_count = 0

    # Apply skip and limit
    if args.skip > 0:
        text_files = text_files[args.skip:]
        print(f"Skipped first {args.skip} files")

    if args.limit:
        text_files = text_files[:args.limit]
        print(f"Limited to {args.limit} files")

    print(f"Processing {len(text_files)} file(s)...\n")

    # Track overall start time
    overall_start_time = time.time()
    recipe_times = []

    for i, text_file in enumerate(text_files, 1):
        recipe_start_time = time.time()

        # Calculate estimated time remaining
        if recipe_times:
            avg_time_per_recipe = sum(recipe_times) / len(recipe_times)
            remaining_recipes = len(text_files) - i + 1
            estimated_remaining = avg_time_per_recipe * remaining_recipes
            print(f"[{i}/{len(text_files)}] Processing: {text_file.name}")
            print(f"  ⏱ Estimated time remaining: {format_time(estimated_remaining)}")
        else:
            print(f"[{i}/{len(text_files)}] Processing: {text_file.name}")

        try:
            # Read the text file
            with open(text_file, 'r', encoding='utf-8') as f:
                text = f.read()

            # Process with LLM
            print(f"  → Extracting recipe data...")
            llm_start = time.time()
            recipe_json, reasoning = process_text_with_llm(client, text, model=args.model)
            llm_time = time.time() - llm_start

            if recipe_json is None:
                error_count += 1
                recipe_times.append(time.time() - recipe_start_time)
                print()
                continue

            print(f"  ✓ Extracted in {format_time(llm_time)}")

            # Check if empty result
            if not recipe_json or recipe_json == {}:
                print(f"  ⚠ No recipe found in text")
                empty_count += 1
                recipe_times.append(time.time() - recipe_start_time)
                print()
                continue

            # Save to output file
            output_filename = text_file.stem + '.json'
            output_path = output_dir / output_filename

            # Handle duplicate filenames
            counter = 1
            original_path = output_path
            while output_path.exists():
                output_path = output_dir / f"{text_file.stem}_{counter}.json"
                counter += 1

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'recipe': recipe_json,
                    'reasoning': reasoning
                }, f, indent=2, ensure_ascii=False)

            recipe_name = recipe_json.get('name', 'Unknown')
            print(f"  ✓ Saved recipe: {recipe_name}")
            print(f"    → {output_path}")

            # Track time
            recipe_total_time = time.time() - recipe_start_time
            recipe_times.append(recipe_total_time)
            print(f"  ⏱ Total time for this recipe: {format_time(recipe_total_time)}")

            success_count += 1

        except Exception as e:
            print(f"  ✗ Error processing file: {e}", file=sys.stderr)
            error_count += 1
            recipe_times.append(time.time() - recipe_start_time)

        print()

    # Summary
    total_elapsed_time = time.time() - overall_start_time

    print("=" * 80)
    print(f"Processing complete!")
    print(f"  ✓ Successfully extracted: {success_count}")
    print(f"  ⚠ No recipe found: {empty_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"  ⏱ Total elapsed time: {format_time(total_elapsed_time)}")

    if recipe_times:
        avg_time = sum(recipe_times) / len(recipe_times)
        print(f"  ⏱ Average time per recipe: {format_time(avg_time)}")

    print(f"\nOutput saved to: {output_dir.absolute()}")


if __name__ == '__main__':
    main()

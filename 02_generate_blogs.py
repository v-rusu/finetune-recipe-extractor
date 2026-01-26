#!/usr/bin/env python3
"""
Generate synthetic recipe data from allrecipes.csv.
For each recipe in the CSV, generates blog post style text with fluff and ads using LLM
"""

from openai import OpenAI
import argparse
import sys
import json
import csv
import time
import random
from pathlib import Path

# Different blog post styles with weights
BLOGPOST_PROMPTS = [
    {
        "name": "instagram_short",
        "weight": 1,
        "prompt": """You are a recipe social media post generator. Create a super short, Instagram-style recipe post.

Format:
- 2-3 sentence description of the recipe and process
- Simple ingredient list (use + or bullets)
- Optional: 1-2 emojis
- Keep it under 200 words total

Be concise and direct. NO long stories, NO fluff, NO advertisements.

Return ONLY the post text."""
    },
    {
        "name": "minimal_organized",
        "weight": 2,
        "prompt": """You are a recipe content generator. Create a clean, minimal recipe post.

Include:
- Brief 1-paragraph introduction
- Ingredient list with measurements
- Step-by-step instructions (numbered)
- Optional: One cooking tip
- Cook time and servings

Keep it under 400 words. Be organized and straightforward.

Return ONLY the recipe text."""
    },
    {
        "name": "fluffy_organized",
        "weight": 5,
        "prompt": """You are a recipe blog post generator. Write a realistic recipe blog post with lots of fluff, personal stories, ads, and all the typical elements found on recipe websites.

Include:
- Long personal stories or anecdotes (2-3 paragraphs)
- ADVERTISEMENT markers scattered throughout
- SEO-optimized descriptions
- Tips and tricks sections
- Detailed ingredient lists with measurements
- Step-by-step instructions
- Metadata like author, date, cook times
- Headings and formatting similar to real recipe blogs

Make it feel authentic and somewhat annoying with too much text before getting to the actual recipe (just like real recipe blogs!).

Return ONLY the blog post text, no JSON or other formatting."""
    },
    {
        "name": "chaotic_unstructured",
        "weight": 2,
        "prompt": """You are generating scraped recipe content that looks like it came from a poorly formatted HTML page.

Create a recipe post that is:
- Somewhat disorganized and rambling
- Has inconsistent formatting
- Mix of paragraphs with ingredients and instructions scattered throughout
- Some repeated information
- Random tangents mid-recipe
- Inconsistent capitalization and punctuation
- Maybe some typos or awkward phrasing
- Still includes the recipe content but in a messy way

Think: badly scraped HTML, poor OCR scan, or auto-translated content.

Return ONLY the messy recipe text."""
    },
    {
        "name": "super_fluffy_chaotic",
        "weight": 1,
        "prompt": """You are a recipe blogger who can't stay on topic. Create an extremely verbose, chaotic recipe post.

Include:
- Multiple long personal stories that barely relate to the recipe
- Constant digressions and tangents
- ADVERTISEMENT markers everywhere
- Repetitive information
- Stream of consciousness writing style
- Inconsistent formatting
- Eventually get to the recipe buried in the text
- Mix ingredients and instructions with personal anecdotes
- Random ALL CAPS for emphasis
- Lots of exclamation points!!!

Make it feel like reading a chaotic food blog that desperately needs an editor.

Return ONLY the blog post text."""
    }
]

def select_blogpost_prompt():
    """Randomly select a blog post prompt based on weights."""
    prompts = BLOGPOST_PROMPTS
    weights = [p["weight"] for p in prompts]
    selected = random.choices(prompts, weights=weights, k=1)[0]
    return selected

def generate_blogpost(client, recipe_data, model):
    """Generate blog post style text from CSV data."""
    # Select a random blog post style
    selected_prompt = select_blogpost_prompt()
    style_name = selected_prompt["name"]
    system_prompt = selected_prompt["prompt"]

    # Prepare user message with recipe details
    user_message = f"""Recipe Information:
Name: {recipe_data['name']}
Category: {recipe_data['group']}
Rating: {recipe_data['rating']}
Number of Raters: {recipe_data['n_rater']}
Number of Reviewers: {recipe_data['n_reviewer']}
Summary: {recipe_data['summary']}
Process/Timing: {recipe_data['process']}
Ingredients: {recipe_data['ingredient']}

Generate the recipe content according to the style specified in the system prompt."""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,  # Higher temperature for more creative blog posts
            max_tokens=6000
        )

        content = response.choices[0].message.content.strip()
        return content, style_name, None

    except Exception as e:
        return None, None, str(e)

def sanitize_filename(name):
    """Convert recipe name to safe filename."""
    # Remove or replace unsafe characters
    safe_name = name.replace('/', '-').replace('\\', '-').replace(':', '-')
    safe_name = safe_name.replace('?', '').replace('*', '').replace('"', '')
    safe_name = safe_name.replace('<', '').replace('>', '').replace('|', '')
    # Limit length
    if len(safe_name) > 200:
        safe_name = safe_name[:200]
    return safe_name

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

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic recipe data from allrecipes.csv',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all recipes in the CSV
  python 02_generate_blogs.py

  # Process only first 10 recipes
  python 02_generate_blogs.py --limit 10

  # Use custom LLM endpoint and model
  python 02_generate_blogs.py --base-url http://localhost:1234/v1 --model llama-3.2-3b-instruct
        """
    )

    parser.add_argument('--csv', default='./kaggle_dataset/allrecipes.csv',
                       help='Path to allrecipes.csv (default: ./kaggle_dataset/allrecipes.csv)')
    parser.add_argument('--output-blog', default='./generated_recipes_blog',
                       help='Output directory for blog posts (default: ./generated_recipes_blog)')
    parser.add_argument('--base-url', default='http://localhost:1234/v1',
                       help='LLM API base URL (default: http://localhost:1234/v1)')
    parser.add_argument('--model', default='unsloth/Qwen3-14B-unsloth-bnb-4bit',
                       help='Model name to use (default: unsloth/Qwen3-14B-unsloth-bnb-4bit)')
    parser.add_argument('--api-key', default='lm-studio',
                       help='API key (default: lm-studio)')
    parser.add_argument('--limit', type=int, default=2,
                       help='Limit number of recipes to process (default: process all)')
    parser.add_argument('--skip', type=int, default=0,
                       help='Skip first N recipes (default: 0)')

    args = parser.parse_args()

    # Setup paths
    csv_path = Path(args.csv)
    output_blog_dir = Path(args.output_blog)

    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' does not exist", file=sys.stderr)
        sys.exit(1)

    # Create output directories
    output_blog_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client
    try:
        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key
        )
        print(f"Connected to LLM at {args.base_url}")
        print(f"Using model: {args.model}\n")
    except Exception as e:
        print(f"Error connecting to LLM: {e}", file=sys.stderr)
        sys.exit(1)

    # Read CSV file - try different encodings if UTF-8 fails
    recipes = []
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    for encoding in encodings:
        try:
            with open(csv_path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    recipes.append(row)
            print(f"Successfully read CSV with {encoding} encoding")
            break
        except UnicodeDecodeError:
            if encoding == encodings[-1]:
                print(f"Error: Could not decode CSV with any standard encoding", file=sys.stderr)
                sys.exit(1)
            continue

    # Apply skip and limit
    if args.skip > 0:
        recipes = recipes[args.skip:]
        print(f"Skipped first {args.skip} recipes")

    if args.limit:
        recipes = recipes[:args.limit]
        print(f"Limited to {args.limit} recipes")

    print(f"Processing {len(recipes)} recipe(s)...\n")

    # Track overall start time
    overall_start_time = time.time()

    # Process each recipe
    success_count = 0
    error_count = 0
    recipe_times = []  # Track time per recipe for estimation
    style_counts = {}  # Track which styles were used

    for i, recipe in enumerate(recipes, 1):
        recipe_start_time = time.time()
        recipe_name = recipe['name']

        # Calculate estimated time remaining
        if recipe_times:
            avg_time_per_recipe = sum(recipe_times) / len(recipe_times)
            remaining_recipes = len(recipes) - i + 1
            estimated_remaining = avg_time_per_recipe * remaining_recipes
            print(f"[{i}/{len(recipes)}] Processing: {recipe_name}")
            print(f"  ⏱ Estimated time remaining: {format_time(estimated_remaining)}")
        else:
            print(f"[{i}/{len(recipes)}] Processing: {recipe_name}")

        # Generate safe filename
        safe_name = sanitize_filename(recipe_name)

        # Generate blog post
        print(f"  → Generating blog post...")
        blog_start = time.time()
        blogpost, style_name, error = generate_blogpost(client, recipe, args.model)
        blog_time = time.time() - blog_start

        if error:
            print(f"  ✗ Blog post generation failed: {error}")
            error_count += 1
            continue

        print(f"  ✓ Blog post generated in {format_time(blog_time)} (style: {style_name})")

        # Save blog post with style metadata
        blog_filename = f"{safe_name}.txt"
        blog_path = output_blog_dir / blog_filename

        # Handle duplicate filenames
        counter = 1
        while blog_path.exists():
            blog_path = output_blog_dir / f"{safe_name}_{counter}.txt"
            counter += 1

        with open(blog_path, 'w', encoding='utf-8') as f:
            # Add metadata header (commented out so it doesn't affect scraping)
            f.write(f"<!-- Generated with style: {style_name} -->\n\n")
            f.write(blogpost)

        print(f"  ✓ Saved blog post: {blog_path.name}")

        # Track style usage
        style_counts[style_name] = style_counts.get(style_name, 0) + 1

        # Calculate total time for this recipe
        recipe_total_time = time.time() - recipe_start_time
        recipe_times.append(recipe_total_time)

        print(f"  ⏱ Total time for this recipe: {format_time(recipe_total_time)}")
        success_count += 1
        print()

    # Summary
    total_elapsed_time = time.time() - overall_start_time

    print("=" * 80)
    print(f"Processing complete!")
    print(f"  ✓ Successfully generated: {success_count}")
    print(f"  ✗ Errors: {error_count}")
    print(f"  ⏱ Total elapsed time: {format_time(total_elapsed_time)}")

    if recipe_times:
        avg_time = sum(recipe_times) / len(recipe_times)
        print(f"  ⏱ Average time per recipe: {format_time(avg_time)}")

    # Show style distribution
    if style_counts:
        print(f"\n  Blog post style distribution:")
        for style, count in sorted(style_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / success_count * 100) if success_count > 0 else 0
            print(f"    • {style}: {count} ({percentage:.1f}%)")

    print(f"Blog posts saved to: {output_blog_dir.absolute()}")

if __name__ == '__main__':
    main()

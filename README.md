# Recipe Extractor Finetuning Pipeline

A pipeline to generate synthetic training data and finetune small language models for recipe extraction from messy blog posts.

## Overview

This project creates a finetuning dataset by:
1. Downloading real recipe data from Kaggle
2. Generating synthetic blog posts with varying styles (fluffy, minimal, chaotic)
3. Using a reasoning model to extract structured recipe JSON with chain-of-thought
4. Combining everything into a finetuning dataset
5. Training a small model (Gemma-3 270M) to extract recipes

## Why This Architecture?

**Separate Requirements**: The pipeline is split into generation scripts (steps 1-4) and finetuning (step 5) with separate requirements files. This allows:
- Running data generation locally on CPU/modest hardware
- Running finetuning for free on Google Colab (with GPU)
- The `finetune.ipynb` notebook can be uploaded directly to Colab

**Why Qwen for Generation?**: The data generation step uses Qwen3-14B because it produces reasoning traces in `<think>` tags. This reasoning style is similar to open-source Chinese reasoning models (like DeepSeek-R1), making the resulting dataset potentially useful for training reasoning models or distilling reasoning capabilities into smaller models.

## Scripts

### 01_download_dataset.py

Downloads the AllRecipes dataset from Kaggle.

| Parameter | Default | Description |
|-----------|---------|-------------|
| (none) | - | No CLI arguments, uses Kaggle credentials |

**Output**: `kaggle_dataset/allrecipes.csv`

**Credentials**: Set `KAGGLE_USERNAME` and `KAGGLE_API_TOKEN` environment variables, or place `kaggle.json` in the project root. Use the Legacy API Credentials.

---

### 02_generate_blogs.py

Generates synthetic blog posts from recipe CSV data using an LLM. Creates varied content styles to simulate real-world recipe scraping scenarios.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--csv` | `./kaggle_dataset/allrecipes.csv` | Input CSV file |
| `--output-blog` | `./generated_recipes_blog` | Output directory for blog posts |
| `--base-url` | `http://localhost:1234/v1` | LLM API endpoint (LM Studio) |
| `--model` | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Model name |
| `--api-key` | `lm-studio` | API key |
| `--limit` | `2` | Number of recipes to process |
| `--skip` | `0` | Skip first N recipes |

**Blog Post Styles** (weighted random selection):
- `instagram_short` (weight: 1) - Super short, Instagram-style
- `minimal_organized` (weight: 2) - Clean, minimal format
- `fluffy_organized` (weight: 5) - Typical recipe blog with stories and ads
- `chaotic_unstructured` (weight: 2) - Poorly formatted, messy
- `super_fluffy_chaotic` (weight: 1) - Extremely verbose and disorganized

---

### 03_generate_recipe_json.py

Extracts structured recipe JSON from blog posts using an LLM with reasoning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input`, `-i` | `./generated_recipes_blog` | Input directory with blog text files |
| `--output`, `-o` | `./llm_output_reasoning` | Output directory for JSON files |
| `--base-url` | `http://localhost:1234/v1` | LLM API endpoint |
| `--model` | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | Model name |
| `--api-key` | `lm-studio` | API key |
| `--limit` | `None` | Process all files |
| `--skip` | `0` | Skip first N files |

**Output Format**: JSON files containing both `recipe` (schema.org Recipe) and `reasoning` (chain-of-thought trace).

---

### 04_generate_finetuning_dataset.py

Combines blog posts and extracted recipes into a finetuning dataset in JSONL format.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--blog-dir` | `./generated_recipes_blog` | Blog posts directory |
| `--llm-dir` | `./llm_output_reasoning` | LLM output JSON directory |
| `--output` | `./finetuning_dataset.jsonl` | Output JSONL file |
| `--system-prompt` | (built-in) | Custom system prompt |
| `--split` | `None` | Train/test split ratio (e.g., 0.8) |
| `--limit` | `None` | Process all examples |

**Output Format**: JSONL with messages array containing system prompt, user (blog text), and assistant (reasoning + JSON) turns.

---

### 05_finetune.py

Finetunes Gemma-3 270M using Unsloth and LoRA.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `./finetuning_dataset.jsonl` | Training dataset |
| `--model-name` | `unsloth/gemma-3-270m-it` | Base model |
| `--max-seq-length` | `8192` | Maximum sequence length |
| `--lora-rank` | `128` | LoRA rank (trainable parameter count) |
| `--batch-size` | `4` | Per-device batch size |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation |
| `--max-steps` | `50` | Training steps (-1 for full epochs) |
| `--num-epochs` | `1` | Epochs (when max-steps is -1) |
| `--learning-rate` | `2e-5` | Learning rate |
| `--output-dir` | `outputs` | Checkpoint directory |
| `--save-dir` | `gemma-3` | Final model save directory |
| `--save-gguf` | `False` | Export to GGUF format |
| `--gguf-quantization` | `Q8_0` | GGUF quantization (Q8_0, BF16, F16) |
| `--load-4bit` | `False` | Load model in 4-bit |
| `--load-8bit` | `False` | Load model in 8-bit |
| `--skip-inference-test` | `False` | Skip post-training inference test |

---

### 06_finetune_wandb.py

Finetunes with Weights & Biases experiment tracking and hyperparameter sweep support.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `./finetuning_dataset.jsonl` | Training dataset |
| `--model-name` | `unsloth/gemma-3-270m-it` | Base model |
| `--max-seq-length` | `8192` | Maximum sequence length |
| `--lora-rank` | `128` | LoRA rank |
| `--lora-alpha` | `(same as rank)` | LoRA alpha scaling |
| `--lora-dropout` | `0` | LoRA dropout |
| `--batch-size` | `4` | Per-device batch size |
| `--gradient-accumulation-steps` | `4` | Gradient accumulation |
| `--max-steps` | `50` | Training steps (-1 for full epochs) |
| `--num-epochs` | `1` | Epochs (when max-steps is -1) |
| `--learning-rate` | `2e-5` | Learning rate |
| `--weight-decay` | `0.001` | Weight decay |
| `--warmup-steps` | `5` | Warmup steps |
| `--lr-scheduler-type` | `linear` | LR scheduler (linear, cosine, constant, constant_with_warmup) |
| `--optimizer` | `adamw_8bit` | Optimizer (adamw_8bit, adamw_torch, sgd, adafactor) |
| `--use-rslora` | `False` | Use rank-stabilized LoRA |
| `--wandb-project` | `recipe-extractor-finetune` | W&B project name |
| `--wandb-entity` | `None` | W&B team/username |
| `--sweep-id` | `None` | Join existing sweep |
| `--sweep-count` | `None` | Number of sweep runs |

---

### sweep.yaml

W&B sweep configuration for hyperparameter optimization using Bayesian search.

**Swept Parameters:**
- `learning_rate`: log-uniform 1e-6 to 1e-4
- `lora_rank`: [32, 64, 128, 256]
- `lora_alpha`: [32, 64, 128, 256]
- `lora_dropout`: [0, 0.05, 0.1]
- `batch_size`: [2, 4, 8]
- `gradient_accumulation_steps`: [2, 4, 8]
- `weight_decay`: log-uniform 1e-4 to 1e-2
- `warmup_steps`: [0, 5, 10, 20]
- `lr_scheduler_type`: [linear, cosine, constant_with_warmup]
- `use_rslora`: [false, true]

## Installation

### For Data Generation (Steps 1-4)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### For Finetuning (Step 5)

**Option A: Local with GPU**
```bash
pip install -r requirements-finetune.txt
```

**Option B: Google Colab (Recommended - Free GPU)**
1. Upload `finetune.ipynb` to Google Colab
2. Upload your `finetuning_dataset.jsonl`
3. Run all cells

## Running the Pipeline

### Step 1: Download Dataset

```bash
# Set Kaggle credentials
export KAGGLE_USERNAME="your_username"
export KAGGLE_API_TOKEN="your_api_token"

# Or place kaggle.json in project root

python 01_download_dataset.py
```

### Step 2: Generate Blog Posts

Requires an LLM server running (e.g., LM Studio with Qwen3-14B).

```bash
# Start LM Studio with Qwen3-14B model first, then:

# Generate a few test blogs
python 02_generate_blogs.py --limit 10

# Generate all blogs (takes a while)
python 02_generate_blogs.py --limit 0  # 0 = no limit
```

### Step 3: Extract Recipe JSON

```bash
# Extract recipes with reasoning traces
python 03_generate_recipe_json.py --limit 10

# Process all
python 03_generate_recipe_json.py
```

### Step 4: Create Finetuning Dataset

```bash
# Create single dataset
python 04_generate_finetuning_dataset.py

# Or create train/test split
python 04_generate_finetuning_dataset.py --split 0.9
```

### Step 5: Finetune

**Local:**
```bash
# Quick test run (50 steps)
python 05_finetune.py

# Full training with GGUF export
python 05_finetune.py --max-steps -1 --num-epochs 1 --save-gguf
```

**Google Colab:**
1. Go to [Google Colab](https://colab.research.google.com/)
2. Upload `finetune.ipynb`
3. Upload `finetuning_dataset.jsonl` to the Colab file browser
4. Select GPU runtime: Runtime > Change runtime type > T4 GPU
5. Run all cells

### Step 5b: Finetune with W&B (Optional)

For experiment tracking and hyperparameter sweeps:

```bash
# Single run with W&B tracking
python 06_finetune_wandb.py --wandb-project my-project

# Create a hyperparameter sweep
wandb sweep sweep.yaml
# Returns: Created sweep with ID: <sweep_id>

# Run sweep agent (executes multiple training runs)
wandb agent <sweep_id>

# Or run via the script with count limit
python 06_finetune_wandb.py --sweep-id <sweep_id> --sweep-count 10
```

## Output

After finetuning, you'll have:
- `gemma-3/` - LoRA adapter weights
- `gemma-3-gguf/` - GGUF model (if `--save-gguf` was used)

The GGUF model can be used with:
- llama.cpp: `llama-cli --model gemma-3-270m-it.Q8_0.gguf -p "Extract recipe from: ..."`
- Ollama: `ollama create recipe-extractor -f ./Modelfile`

## Dataset Format

The finetuning dataset uses a messages format with reasoning traces:

```json
{
  "messages": [
    {"role": "system", "content": "You are a recipe extraction assistant..."},
    {"role": "user", "content": "Extract recipe information from this text:\n\n[blog post]"},
    {"role": "assistant", "content": "<think>\n[reasoning trace]\n</think>\n\n{\"@context\": \"https://schema.org\", ...}"}
  ]
}
```

The reasoning traces are removed during finetuning for non-reasoning models like Gemma-3 270M, but preserved in the dataset for potential use with reasoning-capable models.

## License

MIT

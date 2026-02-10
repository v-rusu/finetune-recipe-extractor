#!/usr/bin/env python3
"""
Recipe Vibes Evaluator

Evaluates how accurately a recipe LD-JSON captures the content and details
from the original blogpost using LLM-based evaluation via OpenRouter.
"""

import json
import os
from typing import Dict, Tuple, Optional

import json_repair

from openai import OpenAI


class VibesEvaluator:
    """Evaluates recipe JSON accuracy against the original blogpost using LLM."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "deepseek/deepseek-v3.2"):
        """
        Initialize the VibesEvaluator.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use for evaluation (default: deepseek/deepseek-v3.2)
        """
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided or set in OPENROUTER_API_KEY env var")
        
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )
        
        self.evaluation_prompt = """You are evaluating how accurately a recipe JSON-LD structure captures the information from an original blog post.

Given:
1. The original blog post text
2. The extracted recipe JSON-LD

Please evaluate the following aspects and provide a score from 0-10 for each:

**Recipe Name Accuracy (0-10):**
- Does the JSON name accurately reflect the recipe name in the blog?
- Is it appropriately formatted?

**Ingredients Accuracy (0-10):**
- Are all ingredients from the blog captured?
- Are quantities and measurements accurate?
- Are ingredient names correctly extracted?

**Instructions Accuracy (0-10):**
- Are all steps from the blog included?
- Is the order correct?
- Are the steps clear and complete?

**Metadata Accuracy (0-10):**
- Are times (prep, cook, total) accurate if provided?
- Is yield/servings correct?
- Are categories and cuisines appropriate?

**Completeness (0-10):**
- Is any important information from the blog missing?
- Are there extra details in the JSON not in the blog?

**Overall Accuracy (0-10):**
- How well does the JSON represent the blog post overall?

Provide your evaluation in the following JSON format:
{
  "scores": {
    "name": <0-10>,
    "ingredients": <0-10>,
    "instructions": <0-10>,
    "metadata": <0-10>,
    "completeness": <0-10>,
    "overall": <0-10>
  },
  "feedback": {
    "name": "<brief explanation>",
    "ingredients": "<brief explanation>",
    "instructions": "<brief explanation>",
    "metadata": "<brief explanation>",
    "completeness": "<brief explanation>",
    "overall": "<brief summary>"
  },
  "issues": [
    "<specific issue 1>",
    "<specific issue 2>"
  ],
  "strengths": [
    "<strength 1>",
    "<strength 2>"
  ]
}

Be thorough but concise in your feedback. Focus on factual accuracy and completeness."""

    def evaluate(
        self, 
        blogpost: str, 
        recipe_json: str,
        temperature: float = 0.3
    ) -> Tuple[Dict, str]:
        """
        Evaluate the recipe JSON against the blogpost.
        
        Args:
            blogpost: The original blog post text
            recipe_json: The extracted recipe in JSON-LD format (as string or dict)
            temperature: Temperature for LLM generation (default: 0.3 for more consistent scoring)
            
        Returns:
            Tuple of (evaluation_dict, raw_response_text)
            evaluation_dict contains scores, feedback, issues, and strengths
        """
        # Convert recipe_json to dict if it's a string
        if isinstance(recipe_json, str):
            try:
                recipe_dict = json_repair.loads(recipe_json)
            except json.JSONDecodeError as e:
                return {
                    "error": f"Invalid JSON provided: {str(e)}",
                    "scores": {"overall": 0}
                }, ""
        else:
            recipe_dict = recipe_json
        
        # Format the recipe JSON nicely for the prompt
        recipe_json_formatted = json.dumps(recipe_dict, indent=2)
        
        # Create the evaluation prompt
        user_message = f"""**BLOG POST:**
```
{blogpost}
```

**EXTRACTED RECIPE JSON:**
```json
{recipe_json_formatted}
```

Please evaluate the accuracy of this JSON extraction."""
        
        # Call the LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.evaluation_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=2000,
            )
            
            raw_response = response.choices[0].message.content
            
            # Try to extract JSON from the response
            evaluation = self._extract_json_from_response(raw_response)
            
            return evaluation, raw_response
            
        except Exception as e:
            return {
                "error": f"API call failed: {str(e)}",
                "scores": {"overall": 0}
            }, ""
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract and parse JSON from the LLM response."""
        # Try to find JSON in code blocks first
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Try to find JSON object directly
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
            else:
                return {"error": "Could not extract JSON from response", "raw": response_text}
        
        try:
            return json_repair.loads(json_str)
        except json.JSONDecodeError:
            return {"error": "Could not parse JSON from response", "raw": response_text}
    
    def evaluate_batch(
        self, 
        blog_recipe_pairs: list[Tuple[str, str]],
        verbose: bool = True
    ) -> list[Dict]:
        """
        Evaluate multiple blog/recipe pairs.
        
        Args:
            blog_recipe_pairs: List of (blogpost, recipe_json) tuples
            verbose: Print progress information
            
        Returns:
            List of evaluation dictionaries
        """
        results = []
        
        for i, (blogpost, recipe_json) in enumerate(blog_recipe_pairs, 1):
            if verbose:
                print(f"Evaluating {i}/{len(blog_recipe_pairs)}...")
            
            evaluation, _ = self.evaluate(blogpost, recipe_json)
            results.append(evaluation)
            
            if verbose and "scores" in evaluation:
                overall = evaluation["scores"].get("overall", 0)
                print(f"  Overall score: {overall}/10")
        
        return results
    
    def print_evaluation(self, evaluation: Dict) -> None:
        """Pretty print an evaluation result."""
        if "error" in evaluation:
            print(f"❌ Error: {evaluation['error']}")
            return
        
        print("\n" + "=" * 70)
        print("VIBES EVALUATION RESULTS")
        print("=" * 70)
        
        scores = evaluation.get("scores", {})
        print("\n📊 SCORES:")
        print(f"  Name:         {scores.get('name', 0)}/10")
        print(f"  Ingredients:  {scores.get('ingredients', 0)}/10")
        print(f"  Instructions: {scores.get('instructions', 0)}/10")
        print(f"  Metadata:     {scores.get('metadata', 0)}/10")
        print(f"  Completeness: {scores.get('completeness', 0)}/10")
        print(f"  ─────────────────────")
        print(f"  OVERALL:      {scores.get('overall', 0)}/10")
        
        feedback = evaluation.get("feedback", {})
        if feedback:
            print("\n💬 FEEDBACK:")
            for key, value in feedback.items():
                if value:
                    print(f"  {key.capitalize()}: {value}")
        
        issues = evaluation.get("issues", [])
        if issues:
            print("\n⚠️  ISSUES:")
            for issue in issues:
                print(f"  • {issue}")
        
        strengths = evaluation.get("strengths", [])
        if strengths:
            print("\n✅ STRENGTHS:")
            for strength in strengths:
                print(f"  • {strength}")
        
        print("=" * 70)


def main():
    """Example usage of the VibesEvaluator."""
    
    # Example blog post and recipe JSON
    example_blog = """
    Today I'm sharing my favorite chocolate chip cookie recipe! These cookies are 
    crispy on the outside and chewy on the inside. You'll need 2 cups of all-purpose 
    flour, 1 cup of butter (softened), 3/4 cup of granulated sugar, 3/4 cup of brown 
    sugar, 2 eggs, 2 teaspoons of vanilla extract, 1 teaspoon of baking soda, 
    1 teaspoon of salt, and 2 cups of chocolate chips.
    
    First, preheat your oven to 375°F (190°C). In a large bowl, cream together the 
    butter and both sugars until fluffy, about 3 minutes. Beat in the eggs one at a 
    time, then stir in the vanilla. In another bowl, whisk together the flour, 
    baking soda, and salt. Gradually blend the dry ingredients into the butter 
    mixture. Fold in the chocolate chips. 
    
    Drop rounded tablespoons of dough onto ungreased cookie sheets, spacing them 
    about 2 inches apart. Bake for 9-11 minutes until golden brown around the edges. 
    Let them cool on the baking sheet for 2 minutes before transferring to a wire rack.
    
    This recipe makes about 48 cookies. Prep time is 15 minutes, baking time is 
    about 10 minutes per batch. Enjoy!
    """
    
    example_json = {
        "@context": "https://schema.org",
        "@type": "Recipe",
        "name": "Classic Chocolate Chip Cookies",
        "description": "Crispy outside, chewy inside chocolate chip cookies",
        "prepTime": "PT15M",
        "cookTime": "PT10M",
        "totalTime": "PT25M",
        "recipeYield": "48 cookies",
        "recipeCategory": "Dessert",
        "recipeIngredient": [
            "2 cups all-purpose flour",
            "1 cup butter, softened",
            "3/4 cup granulated sugar",
            "3/4 cup brown sugar",
            "2 eggs",
            "2 teaspoons vanilla extract",
            "1 teaspoon baking soda",
            "1 teaspoon salt",
            "2 cups chocolate chips"
        ],
        "recipeInstructions": [
            {
                "@type": "HowToStep",
                "text": "Preheat oven to 375°F (190°C)."
            },
            {
                "@type": "HowToStep",
                "text": "Cream together butter and both sugars until fluffy, about 3 minutes."
            },
            {
                "@type": "HowToStep",
                "text": "Beat in eggs one at a time, then stir in vanilla."
            },
            {
                "@type": "HowToStep",
                "text": "In another bowl, whisk together flour, baking soda, and salt."
            },
            {
                "@type": "HowToStep",
                "text": "Gradually blend dry ingredients into butter mixture."
            },
            {
                "@type": "HowToStep",
                "text": "Fold in chocolate chips."
            },
            {
                "@type": "HowToStep",
                "text": "Drop rounded tablespoons of dough onto ungreased cookie sheets, spacing 2 inches apart."
            },
            {
                "@type": "HowToStep",
                "text": "Bake for 9-11 minutes until golden brown around edges."
            },
            {
                "@type": "HowToStep",
                "text": "Cool on baking sheet for 2 minutes, then transfer to wire rack."
            }
        ]
    }
    
    # Check for API key
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("⚠️  OPENROUTER_API_KEY environment variable not set!")
        print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    print("Initializing VibesEvaluator...")
    evaluator = VibesEvaluator()
    
    print("Evaluating example recipe...")
    evaluation, raw_response = evaluator.evaluate(example_blog, example_json)
    
    evaluator.print_evaluation(evaluation)
    
    print("\n📝 Raw LLM Response:")
    print(raw_response)


if __name__ == "__main__":
    main()

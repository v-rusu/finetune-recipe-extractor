"""
Recipe LD-JSON Evaluator

Validates recipe JSON-LD strings against the schema.org Recipe format.
"""

import json
from typing import Dict, List, Tuple, Any

import json_repair

class RecipeEvaluator:
    """Evaluates recipe LD-JSON strings for validity and completeness."""
    
    REQUIRED_FIELDS = [
        "@context",
        "@type",
        "name",
    ]
    
    OPTIONAL_FIELDS = [
        "description",
        "prepTime",
        "cookTime",
        "totalTime",
        "recipeYield",
        "recipeCategory",
        "recipeCuisine",
        "keywords",
        "recipeIngredient",
        "recipeInstructions",
    ]
    
    def __init__(self):
        self.errors = []
        self.warnings = []
    
    def evaluate(self, recipe_json_string: str) -> Tuple[bool, List[str], List[str]]:
        """
        Evaluate a recipe JSON-LD string.
        
        Args:
            recipe_json_string: The recipe in JSON-LD format as a string
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Step 1: Validate JSON syntax
        try:
            recipe_data = json_repair.loads(recipe_json_string)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON syntax: {str(e)}")
            return False, self.errors, self.warnings
        
        # Step 2: Validate structure (must be a dict)
        if not isinstance(recipe_data, dict):
            self.errors.append("Recipe JSON must be an object/dictionary")
            return False, self.errors, self.warnings
        
        # Step 3: Check for required fields
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if field not in recipe_data:
                missing_fields.append(field)
        
        if missing_fields:
            self.errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Step 4: Check for optional fields (warnings only)
        missing_optional = []
        for field in self.OPTIONAL_FIELDS:
            if field not in recipe_data:
                missing_optional.append(field)
        
        if missing_optional:
            self.warnings.append(f"Missing optional fields: {', '.join(missing_optional)}")
        
        # Step 5: Validate field types and values
        self._validate_context(recipe_data.get("@context"))
        self._validate_type(recipe_data.get("@type"))
        self._validate_string_field(recipe_data.get("name"), "name")
        self._validate_string_field(recipe_data.get("description"), "description")
        self._validate_ingredients(recipe_data.get("recipeIngredient"))
        self._validate_instructions(recipe_data.get("recipeInstructions"))
        self._validate_duration(recipe_data.get("prepTime"), "prepTime")
        self._validate_duration(recipe_data.get("cookTime"), "cookTime")
        self._validate_duration(recipe_data.get("totalTime"), "totalTime")
        self._validate_string_field(recipe_data.get("recipeYield"), "recipeYield")
        self._validate_string_field(recipe_data.get("recipeCategory"), "recipeCategory")
        self._validate_string_field(recipe_data.get("recipeCuisine"), "recipeCuisine")
        self._validate_string_field(recipe_data.get("keywords"), "keywords")
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_context(self, value: Any) -> None:
        """Validate @context field."""
        if value is None:
            return  # Already caught by missing fields check
        
        if not isinstance(value, str):
            self.errors.append("@context must be a string")
        elif value != "https://schema.org":
            self.warnings.append(f"@context is '{value}', expected 'https://schema.org'")
    
    def _validate_type(self, value: Any) -> None:
        """Validate @type field."""
        if value is None:
            return  # Already caught by missing fields check
        
        if not isinstance(value, str):
            self.errors.append("@type must be a string")
        elif value != "Recipe":
            self.errors.append(f"@type must be 'Recipe', got '{value}'")
    
    def _validate_string_field(self, value: Any, field_name: str) -> None:
        """Validate a string field."""
        if value is None:
            return  # Already caught by missing fields check
        
        if not isinstance(value, str):
            self.errors.append(f"{field_name} must be a string")
        elif len(value.strip()) == 0:
            self.warnings.append(f"{field_name} is empty")
    
    def _validate_ingredients(self, value: Any) -> None:
        """Validate recipeIngredient field."""
        if value is None:
            return  # Already caught by missing fields check
        
        if not isinstance(value, list):
            self.errors.append("recipeIngredient must be an array")
            return
        
        if len(value) == 0:
            self.warnings.append("recipeIngredient is empty")
            return
        
        for i, ingredient in enumerate(value):
            if not isinstance(ingredient, str):
                self.errors.append(f"recipeIngredient[{i}] must be a string")
    
    def _validate_instructions(self, value: Any) -> None:
        """Validate recipeInstructions field."""
        if value is None:
            return  # Already caught by missing fields check
        
        if not isinstance(value, list):
            self.errors.append("recipeInstructions must be an array")
            return
        
        if len(value) == 0:
            self.warnings.append("recipeInstructions is empty")
            return
        
        for i, instruction in enumerate(value):
            if not isinstance(instruction, dict):
                self.errors.append(f"recipeInstructions[{i}] must be an object")
                continue
            
            # Validate HowToStep structure
            if "@type" not in instruction:
                self.errors.append(f"recipeInstructions[{i}] missing @type field")
            elif instruction["@type"] != "HowToStep":
                self.errors.append(
                    f"recipeInstructions[{i}] @type must be 'HowToStep', got '{instruction['@type']}'"
                )
            
            if "text" not in instruction:
                self.errors.append(f"recipeInstructions[{i}] missing text field")
            elif not isinstance(instruction["text"], str):
                self.errors.append(f"recipeInstructions[{i}] text must be a string")
            elif len(instruction["text"].strip()) == 0:
                self.warnings.append(f"recipeInstructions[{i}] text is empty")
    
    def _validate_duration(self, value: Any, field_name: str) -> None:
        """Validate duration fields (ISO 8601 duration format)."""
        if value is None:
            return  # Already caught by missing fields check
        
        if not isinstance(value, str):
            self.errors.append(f"{field_name} must be a string")
            return
        
        # Basic validation for ISO 8601 duration format (PT prefix)
        if not value.startswith("PT"):
            self.warnings.append(
                f"{field_name} should be in ISO 8601 duration format (e.g., 'PT15M'), got '{value}'"
            )
        elif len(value) <= 2:
            self.warnings.append(f"{field_name} appears to be incomplete: '{value}'")


def main():
    """Example usage of the RecipeEvaluator."""
    evaluator = RecipeEvaluator()
    
    # Example 1: Valid recipe
    valid_recipe = """
    {
      "@context": "https://schema.org",
      "@type": "Recipe",
      "name": "Chocolate Chip Cookies",
      "description": "Delicious homemade chocolate chip cookies",
      "recipeIngredient": ["2 cups flour", "1 cup sugar", "1 cup chocolate chips"],
      "recipeInstructions": [
        {
          "@type": "HowToStep",
          "text": "Mix dry ingredients"
        },
        {
          "@type": "HowToStep",
          "text": "Add chocolate chips and bake"
        }
      ],
      "prepTime": "PT15M",
      "cookTime": "PT12M",
      "totalTime": "PT27M",
      "recipeYield": "24 cookies",
      "recipeCategory": "Dessert",
      "recipeCuisine": "American",
      "keywords": "cookies, dessert, chocolate"
    }
    """
    
    is_valid, errors, warnings = evaluator.evaluate(valid_recipe)
    print("Example 1 - Valid Recipe:")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")
    print()
    
    # Example 2: Invalid recipe (missing fields)
    invalid_recipe = """
    {
      "@context": "https://schema.org",
      "@type": "Recipe",
      "name": "Incomplete Recipe"
    }
    """
    
    is_valid, errors, warnings = evaluator.evaluate(invalid_recipe)
    print("Example 2 - Invalid Recipe (missing fields):")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")
    print()
    
    # Example 3: Invalid JSON
    invalid_json = """
    {
      "@context": "https://schema.org",
      "@type": "Recipe",
      "name": "Broken JSON"
    """
    
    is_valid, errors, warnings = evaluator.evaluate(invalid_json)
    print("Example 3 - Invalid JSON:")
    print(f"  Valid: {is_valid}")
    print(f"  Errors: {errors}")
    print(f"  Warnings: {warnings}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Deterministic JSON repair tool for fixing malformed JSON from LLM outputs.

Applies repairs in a fixed order to ensure consistent, predictable behavior.
Returns detailed metadata about repairs performed.
"""

import json
import re
import argparse
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Union, Any


class RepairType(Enum):
    """Types of repairs that can be performed"""
    MARKDOWN_EXTRACTION = "extracted_from_markdown"
    THINKING_TAGS_REMOVED = "removed_thinking_tags"
    QUOTES_NORMALIZED = "normalized_quotes"
    KEYS_QUOTED = "quoted_unquoted_keys"
    CONSTANTS_FIXED = "fixed_python_constants"
    TRAILING_COMMAS_REMOVED = "removed_trailing_commas"
    DELIMITERS_BALANCED = "balanced_delimiters"
    STRINGS_CLOSED = "closed_unclosed_strings"


@dataclass
class JSONRepairResult:
    """Result of JSON repair operation with metadata"""
    success: bool
    data: Optional[Union[dict, list]] = None
    original: str = ""
    repaired: str = ""
    repairs_made: List[RepairType] = field(default_factory=list)
    error: Optional[str] = None
    parse_attempts: int = 0


class JSONRepairer:
    """
    Deterministic JSON repair engine using multi-pass strategy.

    Repairs are applied in fixed order:
    1. Extract from markdown code blocks
    2. Remove thinking tags
    3. Normalize quotes (single → double)
    4. Quote unquoted keys
    5. Fix Python constants
    6. Remove trailing commas
    7. Close unclosed strings (before balancing!)
    8. Balance delimiters
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        # Repair functions in order of application
        self.repair_pipeline = [
            (RepairType.MARKDOWN_EXTRACTION, self._extract_markdown),
            (RepairType.THINKING_TAGS_REMOVED, self._remove_thinking_tags),
            (RepairType.QUOTES_NORMALIZED, self._normalize_quotes),
            (RepairType.KEYS_QUOTED, self._quote_keys),
            (RepairType.CONSTANTS_FIXED, self._fix_constants),
            (RepairType.TRAILING_COMMAS_REMOVED, self._remove_trailing_commas),
            (RepairType.STRINGS_CLOSED, self._close_strings),
            (RepairType.DELIMITERS_BALANCED, self._balance_delimiters),
        ]

    def repair(self, json_str: str) -> JSONRepairResult:
        """
        Repair malformed JSON string using deterministic strategies.

        Args:
            json_str: Potentially malformed JSON string

        Returns:
            JSONRepairResult with success status, data, and metadata
        """
        result = JSONRepairResult(
            success=False,
            original=json_str,
            repaired=json_str
        )

        # Try parsing original first
        result.parse_attempts += 1
        try:
            data = json.loads(json_str)
            result.success = True
            result.data = data
            return result
        except json.JSONDecodeError:
            pass  # Continue with repairs

        # Apply each repair in sequence
        current = json_str

        for repair_type, repair_func in self.repair_pipeline:
            try:
                repaired = repair_func(current)

                # Track if this repair changed anything
                if repaired != current:
                    result.repairs_made.append(repair_type)
                    current = repaired

                    if self.verbose:
                        print(f"Applied: {repair_type.value}")

                    # Try parsing after each repair
                    result.parse_attempts += 1
                    try:
                        data = json.loads(current)
                        result.success = True
                        result.data = data
                        result.repaired = current
                        return result
                    except json.JSONDecodeError:
                        continue  # Try next repair

            except Exception as e:
                if self.verbose:
                    print(f"Repair {repair_type.value} failed: {e}")
                continue

        # All repairs applied, final parse attempt
        result.repaired = current
        result.parse_attempts += 1
        try:
            data = json.loads(current)
            result.success = True
            result.data = data
        except json.JSONDecodeError as e:
            result.error = str(e)

        return result

    def _extract_markdown(self, s: str) -> str:
        """Extract JSON from markdown code blocks (complete or incomplete)"""
        # Try complete block first: ```json ... ```
        pattern = r'```(?:json)?\s*\n?([\s\S]+?)\n?\s*```'
        match = re.search(pattern, s)
        if match:
            return match.group(1).strip()

        # Handle incomplete block (opening ``` without closing)
        # Pattern: ```json followed by content until end of string
        incomplete_pattern = r'```(?:json)?\s*\n?([\s\S]+)$'
        match = re.search(incomplete_pattern, s)
        if match:
            return match.group(1).strip()

        return s.strip()

    def _remove_thinking_tags(self, s: str) -> str:
        """Remove <think>...</think> tags that some LLMs add"""
        # Remove thinking blocks
        result = re.sub(r'<think>.*?</think>', '', s, flags=re.DOTALL | re.IGNORECASE)
        return result.strip()

    def _normalize_quotes(self, s: str) -> str:
        """
        Convert single quotes to double quotes in a context-aware manner.
        Uses simple FSM to avoid changing quotes inside already-quoted strings.
        """
        class State(Enum):
            NORMAL = 1
            IN_DOUBLE_QUOTE = 2
            IN_SINGLE_QUOTE = 3
            ESCAPED = 4

        result = []
        state = State.NORMAL
        prev_state = State.NORMAL

        for i, char in enumerate(s):
            if state == State.ESCAPED:
                result.append(char)
                state = prev_state
                continue

            if state == State.IN_DOUBLE_QUOTE:
                if char == '\\':
                    prev_state = state
                    state = State.ESCAPED
                    result.append(char)
                elif char == '"':
                    state = State.NORMAL
                    result.append(char)
                else:
                    result.append(char)

            elif state == State.IN_SINGLE_QUOTE:
                if char == '\\':
                    prev_state = state
                    state = State.ESCAPED
                    result.append(char)
                elif char == "'":
                    # Convert closing single quote to double
                    state = State.NORMAL
                    result.append('"')
                else:
                    result.append(char)

            else:  # NORMAL
                if char == '"':
                    state = State.IN_DOUBLE_QUOTE
                    result.append(char)
                elif char == "'":
                    # Convert opening single quote to double
                    state = State.IN_SINGLE_QUOTE
                    result.append('"')
                else:
                    result.append(char)

        return ''.join(result)

    def _quote_keys(self, s: str) -> str:
        """Add quotes around unquoted object keys"""
        # Match pattern: {whitespace}word{whitespace}:
        # Capture word and preserve whitespace
        result = re.sub(
            r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)',
            r'\1"\2"\3',
            s
        )
        return result

    def _fix_constants(self, s: str) -> str:
        """Convert Python constants to JSON equivalents"""
        # Use word boundaries to avoid partial matches
        result = s
        result = re.sub(r'\bTrue\b', 'true', result)
        result = re.sub(r'\bFalse\b', 'false', result)
        result = re.sub(r'\bNone\b', 'null', result)
        return result

    def _remove_trailing_commas(self, s: str) -> str:
        """Remove commas before closing brackets/braces"""
        # Remove comma before } (with optional whitespace)
        result = re.sub(r',(\s*})', r'\1', s)
        # Remove comma before ] (with optional whitespace)
        result = re.sub(r',(\s*])', r'\1', result)
        return result

    def _balance_delimiters(self, s: str) -> str:
        """
        Add missing closing brackets/braces based on opening delimiters.
        Uses stack to track nesting and match types.
        """
        stack = []

        # Track which delimiters are open
        for char in s:
            if char == '{':
                stack.append('{')
            elif char == '[':
                stack.append('[')
            elif char == '}':
                if stack and stack[-1] == '{':
                    stack.pop()
            elif char == ']':
                if stack and stack[-1] == '[':
                    stack.pop()

        # Add missing closers in reverse order
        closers = {'{': '}', '[': ']'}
        closing = ''.join(closers[opener] for opener in reversed(stack))

        return s + closing

    def _close_strings(self, s: str) -> str:
        """
        Close unclosed string literals by adding closing quote at end.
        Handles escaped quotes correctly.
        """
        in_string = False
        escaped = False

        for char in s:
            if escaped:
                escaped = False
                continue

            if char == '\\':
                escaped = True
                continue

            if char == '"':
                in_string = not in_string

        # If we ended while in a string, close it
        if in_string:
            return s + '"'

        return s


def main():
    """CLI interface for JSON repair tool"""
    parser = argparse.ArgumentParser(
        description='Repair malformed JSON from LLM outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Repair JSON from file
  python3 json_repair.py --input broken.json --output fixed.json

  # Repair from stdin
  echo '{"name": "test",}' | python3 json_repair.py

  # Show repair metadata
  python3 json_repair.py --input broken.json --verbose

  # Test mode (validate without saving)
  python3 json_repair.py --input file.json --dry-run
        """
    )

    parser.add_argument('--input', '-i', help='Input JSON file (default: stdin)')
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Show detailed repair information')
    parser.add_argument('--dry-run', action='store_true',
                       help='Validate without saving')
    parser.add_argument('--metadata', action='store_true',
                       help='Output repair metadata as JSON')

    args = parser.parse_args()

    # Read input
    if args.input:
        try:
            with open(args.input, 'r', encoding='utf-8') as f:
                json_str = f.read()
        except FileNotFoundError:
            print(f"Error: File '{args.input}' not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        json_str = sys.stdin.read()

    # Repair
    repairer = JSONRepairer(verbose=args.verbose)
    result = repairer.repair(json_str)

    # Output results
    if args.metadata:
        # Output metadata as JSON
        metadata = {
            "success": result.success,
            "repairs_made": [r.value for r in result.repairs_made],
            "parse_attempts": result.parse_attempts,
            "error": result.error
        }
        print(json.dumps(metadata, indent=2))
        sys.exit(0 if result.success else 1)

    if not result.success:
        print(f"Error: Could not repair JSON: {result.error}", file=sys.stderr)
        if args.verbose:
            print(f"Applied repairs: {[r.value for r in result.repairs_made]}", file=sys.stderr)
            print(f"Parse attempts: {result.parse_attempts}", file=sys.stderr)
        sys.exit(1)

    # Success - output repaired JSON
    if args.verbose:
        print(f"Success! Applied {len(result.repairs_made)} repairs:", file=sys.stderr)
        for repair in result.repairs_made:
            print(f"  - {repair.value}", file=sys.stderr)
        print(f"Parse attempts: {result.parse_attempts}", file=sys.stderr)
        print("", file=sys.stderr)

    if args.dry_run:
        print("Validation successful (dry-run mode)", file=sys.stderr)
        sys.exit(0)

    # Write output
    output_json = json.dumps(result.data, indent=2, ensure_ascii=False)

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(output_json)
            if args.verbose:
                print(f"Wrote repaired JSON to: {args.output}", file=sys.stderr)
        except Exception as e:
            print(f"Error writing file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(output_json)

    sys.exit(0)


if __name__ == '__main__':
    main()

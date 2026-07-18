"""
BharatRAG Utility Helpers
"""

import json


def load_dataset(path: str) -> list:
    """
    Load a BharatRAG benchmark dataset from a JSON file.

    Args:
        path: path to the JSON file

    Returns:
        list of examples
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["data"]


def filter_by_language(examples: list, language: str) -> list:
    """
    Filter examples by language.

    Args:
        examples: list of dataset examples
        language: e.g. "hindi", "marathi", "tamil", "bengali", "telugu", "gujarati", "punjabi", "english"

    Returns:
        filtered list
    """
    return [e for e in examples if e["language"] == language]


def filter_by_domain(examples: list, domain: str) -> list:
    """
    Filter examples by domain.

    Args:
        examples: list of dataset examples
        domain: e.g. "government_scheme", "health", "agriculture"

    Returns:
        filtered list
    """
    return [e for e in examples if e["domain"] == domain]


def pretty_print_results(results: dict) -> None:
    """
    Print BharatRAG evaluation results in a clean format.

    Args:
        results: dict returned by evaluate()
    """
    print("\n📊 BharatRAG Evaluation Results")
    print("─" * 40)
    print(f"  Language:           {results.get('language', 'N/A')}")
    print(f"  Questions scored:   {results.get('num_questions', 'N/A')}")
    print("─" * 40)
    print(f"  Context Relevance:  {results.get('context_relevance', 0)}")
    print(f"  Groundedness:       {results.get('groundedness', 0)}")
    print(f"  Answer Relevance:   {results.get('answer_relevance', 0)}")
    print("─" * 40)
    print(f"  Overall Score:      {results.get('overall', 0)}")

    overall = results.get("overall", 0)
    if overall >= 0.7:
        print("  Rating: 🟢 Good")
    elif overall >= 0.5:
        print("  Rating: 🟡 Moderate")
    else:
        print("  Rating: 🔴 Needs Improvement")
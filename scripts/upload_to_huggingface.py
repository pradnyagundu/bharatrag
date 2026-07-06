"""
Upload BharatRAG benchmark dataset to HuggingFace Hub.

Usage:
    python scripts/upload_to_huggingface.py
    python scripts/upload_to_huggingface.py --data-path data/benchmark.json
    HF_USERNAME=MyUser python scripts/upload_to_huggingface.py

The target repo is <HF_USERNAME>/bharatrag-benchmark, or
defaults to the original maintainer's namespace when unset.
"""

import argparse
import json
import os

from datasets import Dataset


def main():
    parser = argparse.ArgumentParser(
        description="Upload BharatRAG benchmark to HuggingFace"
    )
    parser.add_argument(
        "--data-path",
        default="data/benchmark.json",
        help="Path to benchmark JSON file (default: data/benchmark.json)",
    )
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), args.data_path)

    with open(data_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = raw["data"]

    dataset_dict = {
        "id": [e["id"] for e in examples],
        "language": [e["language"] for e in examples],
        "domain": [e["domain"] for e in examples],
        "question": [e["question"] for e in examples],
        "context": [" | ".join(e["context"]) for e in examples],
        "ground_truth_answer": [e["ground_truth_answer"] for e in examples],
        "hallucinated_answer": [e["hallucinated_answer"] for e in examples],
    }

    dataset = Dataset.from_dict(dataset_dict)
    print(f"Dataset created with {len(dataset)} examples")
    print(dataset)

    hf_user = os.environ.get("HF_USERNAME", "PradnyaGundu")
    repo_id = f"{hf_user}/bharatrag-benchmark"

    dataset.push_to_hub(repo_id)

    print(f"\n✅ Dataset uploaded to {repo_id}")
    print(f"   View at: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()

"""
Upload BharatRAG benchmark dataset to HuggingFace
"""

from datasets import Dataset
import json


# Load your benchmark data
with open("data/benchmark.json", "r", encoding="utf-8") as f:
    raw = json.load(f)

examples = raw["data"]

# Flatten for HuggingFace dataset format
dataset_dict = {
    "id": [e["id"] for e in examples],
    "language": [e["language"] for e in examples],
    "domain": [e["domain"] for e in examples],
    "question": [e["question"] for e in examples],
    "context": [" | ".join(e["context"]) for e in examples],
    "ground_truth_answer": [e["ground_truth_answer"] for e in examples],
    "hallucinated_answer": [e["hallucinated_answer"] for e in examples],
}

# Create HuggingFace dataset
dataset = Dataset.from_dict(dataset_dict)

print(f"Dataset created with {len(dataset)} examples")
print(dataset)

# Push to HuggingFace Hub
dataset.push_to_hub("PradnyaGundu/bharatrag-benchmark")

print("\n✅ Dataset uploaded successfully!")
print("View at: https://huggingface.co/datasets/PradnyaGundu/bharatrag-benchmark")
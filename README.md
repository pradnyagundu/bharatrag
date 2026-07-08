# BharatRAG 🇮🇳

**RAG Evaluation Library for Indian Languages**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/bharatrag.svg)](https://pypi.org/project/bharatrag/)
[![Tests](https://github.com/pradnyagundu/bharatrag/actions/workflows/tests.yml/badge.svg)](https://github.com/pradnyagundu/bharatrag/actions)

BharatRAG is the first open-source RAG evaluation library built specifically for **Indian languages (Hindi, Marathi, Tamil)**.

Existing tools like RAGAS are built and tested on English data. BharatRAG fills the gap — giving developers a reliable way to measure RAG quality in Indic languages, with no API key and no cost.

---

## The Problem

RAG (Retrieval Augmented Generation) systems are being deployed across India for:
- Government scheme chatbots (PM Kisan, Ayushman Bharat)
- Health information systems in regional languages
- EdTech platforms for vernacular learners
- Banking and insurance customer support

But there is **no standard way to evaluate** whether these systems are actually working correctly in Hindi, Marathi, or other Indian languages. RAGAS — the most popular RAG evaluation tool — uses English-first embedding models that produce unreliable scores for Indic text.

**BharatRAG solves this.**

---

## What it measures

BharatRAG computes the **RAG Triad** in Indian languages:

| Metric | Question it answers |
|---|---|
| **Context Relevance** | Did we retrieve the right documents? |
| **Groundedness** | Is the answer based on the context, or hallucinated? |
| **Answer Relevance** | Does the answer actually address the question? |

---

## Installation

```bash
pip install bharatrag
```

---

## Quick Start

```python
from bharatrag import evaluate

results = evaluate(
    questions=["पीएम किसान योजना में कितने रुपये मिलते हैं?"],
    contexts=[[
        "पीएम किसान सम्मान निधि योजना के तहत किसानों को",
        "प्रति वर्ष 6000 रुपये तीन किश्तों में मिलते हैं।"
    ]],
    answers=["पीएम किसान योजना में किसानों को 6000 रुपये मिलते हैं।"],
    language="hindi"
)

print(results)
# {
#   'context_relevance': 0.72,
#   'groundedness': 1.0,
#   'answer_relevance': 0.66,
#   'overall': 0.79,
#   'language': 'hindi',
#   'num_questions': 1
# }
```

### Marathi

```python
results = evaluate(
    questions=["पीएम किसान योजनेत किती रुपये मिळतात?"],
    contexts=[["पीएम किसान सन्मान निधी योजनेंतर्गत शेतकऱ्यांना दरवर्षी 6000 रुपये मिळतात."]],
    answers=["पीएम किसान योजनेत 6000 रुपये मिळतात."],
    language="marathi"
)
```

### Tamil

```python
results = evaluate(
    questions=["பிஎம் கிசான் திட்டத்தில் எவ்வளவு பணம் கிடைக்கிறது?"],
    contexts=[["பிஎம் கிசான் திட்டத்தின் கீழ் விவசாயிகளுக்கு ஆண்டுக்கு 6000 ரூபாய் கிடைக்கிறது."]],
    answers=["பிஎம் கிசான் திட்டத்தில் 6000 ரூபாய் கிடைக்கிறது."],
    language="tamil"
)
```

### Individual metrics

```python
from bharatrag.metrics.context_relevance import ContextRelevance

cr = ContextRelevance(language="hindi")
score = cr.score(
    question="भारत की राजधानी क्या है?",
    contexts=["भारत की राजधानी नई दिल्ली है।"]
)
print(score)  # 0.61
```

---

## Framework Integrations

BharatRAG plugs directly into LangChain and LlamaIndex — evaluate Indic RAG systems inside your existing pipelines.

### LangChain

```bash
pip install bharatrag[langchain]
```

```python
from bharatrag.integrations import BharatRAGLangChainEvaluator

evaluator = BharatRAGLangChainEvaluator(metric="groundedness", language="hindi")

result = evaluator.evaluate_strings(
    prediction="पीएम किसान योजना में 6000 रुपये मिलते हैं।",
    reference="प्रधानमंत्री किसान सम्मान निधि योजना के तहत किसानों को 6000 रुपये मिलते हैं।",
    input="पीएम किसान योजना में कितने रुपये मिलते हैं?"
)
print(result)  # {'score': 1.0}
```

### LlamaIndex

```bash
pip install bharatrag[llamaindex]
```

```python
from bharatrag.integrations import BharatRAGLlamaIndexEvaluator

evaluator = BharatRAGLlamaIndexEvaluator(metric="overall", language="hindi")

result = evaluator.evaluate(
    query="पीएम किसान योजना में कितने रुपये मिलते हैं?",
    contexts=["प्रधानमंत्री किसान सम्मान निधि योजना के तहत किसानों को 6000 रुपये मिलते हैं।"],
    response="पीएम किसान योजना में 6000 रुपये मिलते हैं।"
)
print(result.score)
```

---

## Supported Languages

| Language | Embedding Model |
|---|---|
| Hindi | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Marathi | `l3cube-pune/marathi-sentence-bert-nli` |
| Tamil | `l3cube-pune/tamil-sentence-bert-nli` |
| English | `sentence-transformers/all-MiniLM-L6-v2` |

More languages coming soon — Bengali, Gujarati, Punjabi.

---

## Benchmark Dataset

BharatRAG ships with a hand-curated benchmark dataset of **70 QA pairs** across Hindi, Marathi, and Tamil, spanning:
- Government schemes (PM Kisan, Ayushman Bharat, Jan Dhan, Ujjwala)
- Agriculture (crop insurance, drip irrigation, organic farming)
- Health (diabetes, TB, anaemia, sanitation)
- Education (Mid Day Meal, Beti Bachao, NEP 2020)
- Banking & Finance (UPI, KYC, net banking)

Each example includes a correct answer and a hallucinated answer for evaluation testing.

- **Location in repo:** `data/benchmark.json`
- **On HuggingFace:** [PradnyaGundu/bharatrag-benchmark](https://huggingface.co/datasets/PradnyaGundu/bharatrag-benchmark)

---

## Why BharatRAG?

| Feature | RAGAS | BharatRAG |
|---|---|---|
| English RAG evaluation | ✅ | ✅ |
| Hindi RAG evaluation | ❌ Unreliable | ✅ |
| Marathi / Tamil evaluation | ❌ Not supported | ✅ |
| Indic benchmark dataset | ❌ | ✅ |
| LangChain / LlamaIndex integration | ✅ | ✅ |
| Free, no API key needed | ❌ (needs LLM judge) | ✅ Fully offline |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Benchmarks

BharatRAG is benchmarked against public Indic QA datasets (IndicQA, XQuAD)
with optional comparison to RAGAS and DeepEval.

📊 [View the latest benchmark report →](docs/benchmarks/latest_report.md)

Run benchmarks locally:

```bash
pip install bharatrag[benchmarks]
python scripts/benchmark_public_datasets.py --languages hindi tamil
```

---

## Roadmap

- [x] Hindi support
- [x] Marathi support
- [x] Tamil support
- [x] 70-example benchmark dataset
- [x] LangChain integration
- [x] LlamaIndex integration
- [ ] Bengali, Gujarati, Punjabi support
- [ ] Streamlit UI for interactive evaluation
- [ ] Hinglish / code-switching support
- [x] Benchmarking vs RAGAS / DeepEval
- [ ] Expand benchmark dataset to 500+ examples

---

## Contributors

Huge thanks to the community contributors who've helped shape BharatRAG:

- [@rishabh-108272](https://github.com/rishabh-108272) — LangChain & LlamaIndex integrations, groundedness bug fix
- [@AshayK003](https://github.com/AshayK003) — CI improvements, model caching, logging, dependency cleanup
- [@Yashwanth-Kumar-Kotla](https://github.com/Yashwanth-Kumar-Kotla) — language-agnostic benchmark runner

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Author

**Pradnya Gundu**
B.E. Artificial Intelligence & Data Science, APCOER Pune

- GitHub: [@pradnyagundu](https://github.com/pradnyagundu)
- LinkedIn: [pradnya-gundu](https://linkedin.com/in/pradnya-gundu-b28737249)

---

## License

MIT License — free to use, modify, and distribute.
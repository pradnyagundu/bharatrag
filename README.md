# bharatrag
Open-source RAG evaluation library for Indian languages (Hindi, Marathi). pip install bharatrag
# BharatRAG 🇮🇳

**RAG Evaluation Library for Indian Languages**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/pradnyagundu/bharatrag/actions/workflows/tests.yml/badge.svg)](https://github.com/pradnyagundu/bharatrag/actions/workflows/tests.yml)
[![GitHub](https://img.shields.io/badge/GitHub-bharatrag-black.svg)](https://github.com/pradnyagundu/bharatrag)

BharatRAG is the first open-source RAG evaluation library built specifically for **Indian languages (Hindi and Marathi)**. 

Existing tools like RAGAS are built and tested on English data. BharatRAG fills the gap — giving developers a reliable way to measure RAG quality in Indic languages.

---

## The Problem

RAG (Retrieval Augmented Generation) systems are being deployed across India for:
- Government scheme chatbots (PM Kisan, Ayushman Bharat)
- Health information systems in regional languages
- EdTech platforms for vernacular learners
- Banking and insurance customer support

But there is **no standard way to evaluate** whether these systems are actually working correctly in Hindi, Marathi, or other Indian languages.

RAGAS — the most popular RAG evaluation tool — uses English-first embedding models that produce unreliable scores for Indic text.

**BharatRAG solves this.**

---

## What it measures

BharatRAG computes the **RAG Triad** in Hindi and Marathi:

| Metric | Question it answers |
|---|---|
| **Context Relevance** | Did we retrieve the right documents? |
| **Groundedness** | Is the answer based on the context, or hallucinated? |
| **Answer Relevance** | Does the answer actually address the question? |

---

## Benchmark Results

BharatRAG evaluated on 20 Hindi + Marathi QA pairs across government schemes, health, and agriculture domains:

| Metric | Hindi ✅ Correct | Hindi ❌ Hallucinated | Marathi ✅ Correct | Marathi ❌ Hallucinated |
|---|---|---|---|---|
| Context Relevance | 0.4793 | 0.4793 | 0.4327 | 0.4327 |
| Groundedness | 0.9167 | 0.7000 | 0.9667 | 0.2000 |
| Answer Relevance | 0.6221 | 0.5417 | 0.5072 | 0.2959 |
| **Overall** | **0.6727** | **0.5737** | **0.6355** | **0.3095** |

BharatRAG correctly scores hallucinated answers **lower** than correct answers in both languages.
Marathi hallucination detection shows a **2x difference** in overall score (0.6355 vs 0.3095).

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

### Marathi support

```python
results = evaluate(
    questions=["पीएम किसान योजनेत किती रुपये मिळतात?"],
    contexts=[[
        "पीएम किसान सन्मान निधी योजनेंतर्गत शेतकऱ्यांना",
        "दरवर्षी 6000 रुपये तीन हप्त्यांमध्ये मिळतात."
    ]],
    answers=["पीएम किसान योजनेत 6000 रुपये मिळतात."],
    language="marathi"
)
```

### Individual metrics

```python
from bharatrag.metrics.context_relevance import ContextRelevance
from bharatrag.metrics.groundedness import Groundedness
from bharatrag.metrics.answer_relevance import AnswerRelevance

cr = ContextRelevance(language="hindi")
score = cr.score(
    question="भारत की राजधानी क्या है?",
    contexts=["भारत की राजधानी नई दिल्ली है।"]
)
print(score)  # 0.61
```

---

## Supported Languages

| Language | Model Used |
|---|---|
| Hindi | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Marathi | `l3cube-pune/marathi-sentence-bert-nli` |
| English | `sentence-transformers/all-MiniLM-L6-v2` |

More languages coming soon — Tamil, Bengali, Gujarati.

---

## Benchmark Dataset

BharatRAG ships with a hand-curated benchmark dataset of **20 Hindi + Marathi QA pairs** across:
- Government schemes (PM Kisan, Ayushman Bharat, Jan Dhan, Ujjwala)
- Health (diabetes, sanitation)
- Agriculture (wheat sowing, crop insurance)
- Education (Mid Day Meal, Beti Bachao)

Each example includes a correct answer and a hallucinated answer for evaluation testing.

Dataset location: `data/benchmark.json`

---

## Project Structure

bharatrag/
├── bharatrag/
│   ├── init.py          # evaluate() function
│   ├── embeddings/
│   │   └── indic_embeddings.py   # Indic embedding models
│   └── metrics/
│       ├── context_relevance.py  # Metric 1
│       ├── groundedness.py       # Metric 2
│       └── answer_relevance.py   # Metric 3
├── data/
│   └── benchmark.json       # 20 Hindi+Marathi QA pairs
├── tests/
│   └── test_metrics.py      # 21 pytest tests
└── examples/
└── run_benchmark.py     # Benchmark runner

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Tests run automatically on every PR via [GitHub Actions](https://github.com/pradnyagundu/bharatrag/actions/workflows/tests.yml).

---

## Why BharatRAG?

| Feature | RAGAS | BharatRAG |
|---|---|---|
| English RAG evaluation | ✅ | ✅ |
| Hindi RAG evaluation | ❌ Unreliable | ✅ |
| Marathi RAG evaluation | ❌ Not supported | ✅ |
| Indic benchmark dataset | ❌ | ✅ |
| Free, no API key needed | ✅ | ✅ |

---

## Roadmap

- [x] Hindi support
- [x] Marathi support
- [x] 20-example benchmark dataset
- [ ] Tamil support
- [ ] Bengali support
- [ ] 100-example benchmark dataset
- [ ] LangChain integration
- [ ] LlamaIndex integration
- [ ] HuggingFace Spaces demo

---

## Author

**Pradnya Gundu**
B.E. Artificial Intelligence & Data Science, APCOER Pune

- GitHub: [@pradnyagundu](https://github.com/pradnyagundu)
- LinkedIn: [pradnya-gundu](https://linkedin.com/in/pradnya-gundu-b28737249)

---

## License

MIT License — free to use, modify, and distribute.

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, code style, and what we're looking for.

Especially looking for:
- New Indian language support
- More benchmark QA pairs
- Integration with LangChain/LlamaIndex
- Test coverage improvements

Open an [issue](https://github.com/pradnyagundu/bharatrag/issues) or submit a PR on GitHub.
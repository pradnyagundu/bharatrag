# BharatRAG 🇮🇳

**RAG Evaluation Library for Indian Languages**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/bharatrag.svg)](https://pypi.org/project/bharatrag/)
[![Tests](https://github.com/pradnyagundu/bharatrag/actions/workflows/tests.yml/badge.svg)](https://github.com/pradnyagundu/bharatrag/actions)

BharatRAG is the first open-source RAG evaluation library built specifically for **Indian languages (Hindi, Marathi, Tamil, Bengali, Telugu, Gujarati, Punjabi)**.

Existing tools like RAGAS are built and tested on English data. BharatRAG fills the gap — giving developers a reliable way to measure RAG quality in Indic languages, with no API key and no cost.

> **BharatRAG** was created and is maintained by **[Pradnya Gundu](https://github.com/pradnyagundu)** — original author and project lead. First released July 4, 2026.

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

## Running the Streamlit Dashboard

The optional dashboard provides an interactive way to run BharatRAG evaluations,
explore the bundled benchmark, and compare language-level results.

```bash
git clone https://github.com/pradnyagundu/bharatrag.git
cd bharatrag
pip install -e ".[dashboard]"
streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit. The first evaluation for a language may
take longer because its embedding model is loaded locally. The dashboard needs
no API key and keeps manual input in the local Streamlit process.

If PyPI is unavailable but the dashboard dependencies are already installed,
avoid dependency resolution and install the source checkout directly:

```bash
python -m pip install --no-build-isolation --no-deps -e .
python -m streamlit run streamlit_app.py
```

Use this constrained-network fallback only after confirming that `streamlit` and
`plotly` are available in the same Python environment.

The dashboard adds two optional dependencies only:

- `streamlit` for the application interface
- `plotly` for interactive charts

### Dashboard tour

- **Evaluation:** Choose a supported language, then evaluate one example or a
  batch. Four status-aware metric cards and expandable per-sample details make
  weak groundedness easy to spot.
- **Analytics:** View a metric comparison bar chart, a score histogram for
  multi-sample runs, and a RAG-triad radar chart.
- **Benchmark Explorer:** Filter the 120 bundled records by language and browse
  their question, retrieved context, correct answer, and hallucinated answer.
- **Language Comparison:** Run the benchmark across its supported languages to
  compare correct and hallucinated answer averages in a grouped bar chart.

The dashboard is intentionally a presentation layer: all scores are produced by
the existing `bharatrag.evaluate` API and BharatRAG metric classes.

### Interface design system

The dashboard uses a unified, accessibility-conscious visual system designed
for professional AI tooling. It pairs an Indigo primary action colour with
semantic Green, Amber, and Red score states; white surfaces; an Indigo focus
ring; and a dark Slate navigation sidebar. The Streamlit theme lives in
`.streamlit/config.toml`, while shared CSS and Plotly tokens are maintained in
`dashboard/theme.py`. This keeps native Streamlit controls, metric cards, and
analytics visually consistent without changing any evaluation workflow.

### Example screenshot descriptions

1. **Manual evaluation workspace:** a dark navigation sidebar sits beside a
   gradient BharatRAG header, question/context/answer inputs, and four white
   score cards with coloured status chips and progress bars.
2. **Benchmark comparison:** the correct and hallucinated answer variants are
   shown in one metric table, with a grouped Plotly chart and radar profile
   making groundedness differences immediately visible.
3. **Benchmark explorer:** language-filtered benchmark samples present their
   context chunks alongside clearly separated correct and hallucinated answers,
   with Previous and Next controls for a polished demo flow.

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

### Telugu

```python
results = evaluate(
    questions=["పీఎం కిసాన్ పథకంలో రైతులకు ఎంత డబ్బు లభిస్తుంది?"],
    contexts=[["ప్రధానమంత్రి కిసాన్ సమ్మాన్ నిధి పథకం కింద రైతులకు సంవత్సరానికి 6000 రూపాయలు లభిస్తుంది."]],
    answers=["పీఎం కిసాన్ పథకంలో 6000 రూపాయలు లభిస్తాయి."],
    language="telugu"
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
| Bengali | `l3cube-pune/bengali-sentence-bert-nli` |
| Telugu | `l3cube-pune/telugu-sentence-bert-nli` |
| Gujarati | `l3cube-pune/gujarati-sentence-bert-nli` |
| Punjabi | `l3cube-pune/punjabi-sentence-bert-nli` |
| English | `sentence-transformers/all-MiniLM-L6-v2` |

More languages coming soon — Hinglish (code-switching).

---

## Benchmark Dataset

BharatRAG ships with a hand-curated benchmark dataset of **150 QA pairs** across Hindi, Marathi, Tamil, Bengali, Telugu, Gujarati, and Punjabi, spanning:
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
| Marathi / Tamil / Bengali / Telugu / Gujarati / Punjabi evaluation | ❌ Not supported | ✅ |
| Indic benchmark dataset | ❌ | ✅ |
| LangChain / LlamaIndex integration | ✅ | ✅ |
| Free, no API key needed | ❌ (needs LLM judge) | ✅ Fully offline |

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

By default, pytest runs the fast suite only. These tests use deterministic
fake embedders and do not download embedding models. To run model-dependent
integration tests explicitly:

```bash
pytest tests/ -m "integration" -v
```

---

## Roadmap

- [x] Hindi support
- [x] Marathi support
- [x] Tamil support
- [x] Bengali support
- [x] Telugu support
- [x] Gujarati support
- [x] Punjabi support
- [x] 150-example benchmark dataset (Hindi, Marathi, Tamil, Bengali, Telugu, Gujarati, Punjabi)
- [x] LangChain integration
- [x] LlamaIndex integration
- [x] Streamlit UI for interactive evaluation
- [ ] Hinglish / code-switching support
- [ ] Benchmarking vs RAGAS / DeepEval
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

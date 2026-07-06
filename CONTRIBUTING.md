# Contributing to BharatRAG

Thanks for considering contributing! This guide covers how to get started.

## Quick Start

```bash
# Clone and install
git clone https://github.com/pradnyagundu/bharatrag.git
cd bharatrag
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

## Making Changes

1. **Create a branch** from `main`:
   ```bash
   git checkout -b fix/your-fix-name
   ```

2. **Make focused commits**. Each commit should do one thing.

3. **Keep the diff small**. Prefer surgical changes over large refactors.

4. **Add tests** for any new functionality. We use pytest.

5. **Run the tests** before submitting:
   ```bash
   pytest tests/ -v
   ```

## Code Style

- Follow PEP 8
- Use type hints on public functions
- Write Google-style docstrings (Args / Returns / Example)
- Keep functions focused and small
- Use `logging` instead of `print()` in library code

## What We Welcome

| Contribution | Difficulty | Good First Issue? |
|---|---|---|
| New Indian language (Tamil, Bengali, Gujarati) | Medium | No |
| More benchmark QA pairs | Easy | **Yes** |
| Bug fixes | Easy–Medium | **Yes** |
| Documentation improvements | Easy | **Yes** |
| Test coverage | Easy | **Yes** |
| LangChain / LlamaIndex integration | Medium | No |
| Streamlit demo app | Medium | No |

## Adding a New Language

1. Add the language code and a SentenceTransformer model name to `INDIC_MODELS` in `bharatrag/embeddings/indic_embeddings.py`.
2. Add a language key to `_SUPPORTED_LANGUAGES` in `bharatrag/__init__.py`.
3. Add at least 5 QA pairs to `data/benchmark.json`.
4. Run the benchmark: `python examples/run_benchmark.py`.
5. Update the README benchmark table.

## Dataset Contributions

The benchmark dataset lives at `data/benchmark.json`. We welcome:

- More QA pairs for existing languages
- New domains (banking, insurance, legal, education)
- Edge cases (mixed-language Hinglish, numerical questions)

Format per entry:

```json
{
  "id": "health-005",
  "language": "hindi",
  "domain": "health",
  "question": "...",
  "context": ["..."],
  "ground_truth_answer": "...",
  "hallucinated_answer": "..."
}
```

## Opening a PR

1. Push your branch to your fork.
2. Open a PR against `pradnyagundu/bharatrag:main`.
3. In the PR description, explain:
   - What the change does
   - Why it's needed
   - How you tested it
4. The CI will run automatically.

## Questions?

Open a [GitHub Issue](https://github.com/pradnyagundu/bharatrag/issues) for questions or ideas.

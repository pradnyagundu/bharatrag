"""
BharatRAG CLI — Evaluate RAG systems from the command line.

Usage:
    bharatrag evaluate --data eval.json --language hindi
    bharatrag evaluate --data eval.json --language gujarati --output results.json
    bharatrag languages
    bharatrag --version

Design principle: This CLI never hardcodes anything that lives in the core library.
All dynamic data (supported languages, metrics, version) is sourced at runtime
from the library itself, so the CLI stays in sync automatically with any updates.
"""

import argparse
import json
import sys
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from importlib.metadata import version as _metadata_version
    __version__ = _metadata_version("bharatrag")
except Exception:
    __version__ = "0.0.0"


# ── Dynamic library introspection ─────────────────────────────────────────────
# These functions read data directly from the library registries at runtime.
# If a new language or metric is added to the core, the CLI reflects it
# immediately with ZERO changes required here.

def _get_supported_languages() -> tuple:
    """
    Read supported languages directly from indic_embeddings.INDIC_MODELS.
    Safe to call at any time — importing indic_embeddings no longer loads
    PyTorch (SentenceTransformer is now lazily imported inside _load_model).
    """
    from bharatrag.embeddings.indic_embeddings import SUPPORTED_LANGUAGES
    return SUPPORTED_LANGUAGES


def _get_evaluate_fn():
    """
    Lazily import the evaluate() function from the core library.
    Any new parameters added to evaluate() are automatically available.
    """
    from bharatrag import evaluate
    return evaluate


# ── ANSI colour helpers ────────────────────────────────────────────────────────
def _supports_color() -> bool:
    """Return True if the terminal supports ANSI colours."""
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    if _supports_color():
        return f"\033[{code}m{text}\033[0m"
    return text


def bold(t):    return _c(t, "1")
def cyan(t):    return _c(t, "96")
def green(t):   return _c(t, "92")
def yellow(t):  return _c(t, "93")
def red(t):     return _c(t, "91")
def dim(t):     return _c(t, "2")


# ── Banner ─────────────────────────────────────────────────────────────────────
def _make_banner():
    return f"""
    ____  __                     __  ____  ___   ______
   / __ )/ /_  ____ __________ _/ /_/ __ \\/   | / ____/
  / __  / __ \\/ __ `/ ___/ __ `/ __/ /_/ / /| |/ / __  
 / /_/ / / / / /_/ / /  / /_/ / /_/ _, _/ ___ / /_/ /  
/_____/_/ /_/\\__,_/_/   \\__,_/\\__/_/ |_/_/  |_\\____/   
                                                       
  RAG Evaluation for Indian Languages  •  v{__version__}"""


def _print_banner():
    print(cyan(_make_banner()))
    print()


# ── Subcommand: languages ──────────────────────────────────────────────────────
def cmd_languages(_args):
    """
    Lists all supported languages.
    Dynamically sourced from indic_embeddings.INDIC_MODELS — no hardcoding.
    """
    _print_banner()
    languages = _get_supported_languages()
    print(bold(f"Supported Languages  ({len(languages)} total)"))
    print(dim("─" * 40))
    for lang in languages:
        print(f"  {green('✓')}  {lang}")
    print()


# ── Subcommand: evaluate ───────────────────────────────────────────────────────
def cmd_evaluate(args):
    _print_banner()

    # ── Validate language against the live registry ────────────────────────────
    # This happens at evaluation time (not at import time) to keep startup fast.
    supported = _get_supported_languages()
    language = args.language.lower()
    if language not in supported:
        print(red(f"✗ Unsupported language '{language}'."))
        print(dim(f"  Supported: {', '.join(supported)}"))
        print(dim("  Run 'bharatrag languages' to see the full list."))
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────────
    data_path = args.data
    if not os.path.isfile(data_path):
        print(red(f"✗ File not found: {data_path}"))
        sys.exit(1)

    print(bold("Loading dataset:"), cyan(data_path))
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as exc:
        print(red(f"✗ Invalid JSON: {exc}"))
        sys.exit(1)

    # Accept either the full benchmark envelope OR a plain list
    if isinstance(raw, dict) and "data" in raw:
        examples = raw["data"]
    elif isinstance(raw, list):
        examples = raw
    else:
        print(red("✗ Unrecognised JSON format. Expected a list or a dict with a 'data' key."))
        sys.exit(1)

    # Filter by language
    examples = [e for e in examples if e.get("language", "").lower() == language]

    if not examples:
        print(yellow(f"⚠  No examples found for language '{language}'."))
        sys.exit(1)

    # ── Build inputs ───────────────────────────────────────────────────────────
    questions, contexts, answers = [], [], []
    for ex in examples:
        q = ex.get("question")
        ctx = ex.get("context")
        ans = ex.get("ground_truth_answer") or ex.get("answer")
        if not q or ctx is None or not ans:
            continue
        if isinstance(ctx, str):
            ctx = [ctx]
        questions.append(q)
        contexts.append(ctx)
        answers.append(ans)

    if not questions:
        print(red("✗ No usable examples found (missing question/context/answer fields)."))
        sys.exit(1)

    print(bold("Language:"), cyan(language))
    print(bold("Examples:"), cyan(str(len(questions))))
    print()

    # ── Run evaluation ─────────────────────────────────────────────────────────
    # We fetch evaluate() lazily here — so any new parameters added upstream
    # are automatically available without changing the CLI.
    print(dim("─" * 44))
    print(bold("Running evaluation…"))
    print(dim("─" * 44))

    try:
        evaluate = _get_evaluate_fn()
        results = evaluate(
            questions=questions,
            contexts=contexts,
            answers=answers,
            language=language,
        )
    except Exception as exc:
        print(red(f"✗ Evaluation failed: {exc}"))
        sys.exit(1)

    # ── Pretty-print results ───────────────────────────────────────────────────
    # Results are rendered from the dict returned by evaluate(), so any new
    # metrics added to evaluate()'s output will be automatically displayed.
    _print_results(results)

    # ── Optionally write JSON output ───────────────────────────────────────────
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print()
            print(green(f"✓ Results saved to: {args.output}"))
        except OSError as exc:
            print(yellow(f"⚠  Could not write output: {exc}"))


def _print_results(results: dict):
    """
    Renders evaluation results.
    The three core metrics are displayed explicitly with progress bars.
    Any additional keys returned by evaluate() are printed automatically below,
    so new metrics added upstream appear in the CLI output with zero changes here.
    """
    KNOWN_DISPLAY_KEYS = {
        "context_relevance", "groundedness", "answer_relevance",
        "overall", "language", "num_questions",
    }

    overall = results.get("overall", 0)

    if overall >= 0.7:
        rating = green("🟢 Good")
        bar_colour = green
    elif overall >= 0.5:
        rating = yellow("🟡 Moderate")
        bar_colour = yellow
    else:
        rating = red("🔴 Needs Improvement")
        bar_colour = red

    def _bar(score: float, width: int = 20) -> str:
        filled = round(score * width)
        bar = "█" * filled + "░" * (width - filled)
        return bar_colour(f"[{bar}]")

    print()
    print(bold("📊 BharatRAG Evaluation Results"))
    print(dim("─" * 44))
    print(f"  {bold('Language:')}          {cyan(results.get('language', 'N/A'))}")
    print(f"  {bold('Questions scored:')}  {cyan(str(results.get('num_questions', 'N/A')))}")
    print(dim("─" * 44))

    cr = results.get("context_relevance", 0)
    gr = results.get("groundedness", 0)
    ar = results.get("answer_relevance", 0)

    print(f"  {bold('Context Relevance:')}  {_bar(cr)} {cyan(str(cr))}")
    print(f"  {bold('Groundedness:')}       {_bar(gr)} {cyan(str(gr))}")
    print(f"  {bold('Answer Relevance:')}   {_bar(ar)} {cyan(str(ar))}")

    # ── Auto-display any NEW metrics added to evaluate() in the future ─────────
    extra_metrics = {
        k: v for k, v in results.items()
        if k not in KNOWN_DISPLAY_KEYS and isinstance(v, (int, float))
    }
    for key, val in extra_metrics.items():
        label = key.replace("_", " ").title()
        print(f"  {bold(label + ':')}       {_bar(val)} {cyan(str(val))}")

    print(dim("─" * 44))
    print(f"  {bold('Overall Score:')}      {_bar(overall)} {cyan(str(overall))}")
    print(f"  {bold('Rating:')}             {rating}")
    print()


# ── Argument parser ────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bharatrag",
        description="BharatRAG — RAG Evaluation CLI for Indian Languages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  bharatrag evaluate --data data/benchmark.json --language hindi\n"
            "  bharatrag evaluate --data eval.json --language gujarati --output results.json\n"
            "  bharatrag languages\n"
        ),
    )
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # ── evaluate ────────────────────────────────────────────────────────────
    ev = subparsers.add_parser(
        "evaluate",
        help="Run RAG evaluation on a JSON dataset",
        description="Evaluate a RAG system using BharatRAG metrics.",
    )
    ev.add_argument(
        "--data", "-d",
        required=True,
        metavar="PATH",
        help="Path to the JSON dataset file",
    )
    ev.add_argument(
        "--language", "-l",
        required=True,
        metavar="LANG",
        # NOTE: No 'choices=' here on purpose.
        # Validation is done lazily in cmd_evaluate() against the live registry
        # so that --help stays instant and new languages work automatically.
        help="Target language (run 'bharatrag languages' to see all supported options)",
    )
    ev.add_argument(
        "--output", "-o",
        default=None,
        metavar="PATH",
        help="Optional path to save results as JSON",
    )
    ev.set_defaults(func=cmd_evaluate)

    # ── languages ──────────────────────────────────────────────────────────
    lang_p = subparsers.add_parser(
        "languages",
        help="List all supported languages (auto-updates when new languages are added)",
    )
    lang_p.set_defaults(func=cmd_languages)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()

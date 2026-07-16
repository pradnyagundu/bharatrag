"""Interactive Streamlit dashboard for BharatRAG evaluations.

Run from the repository root with:

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Sequence

import streamlit as st

from dashboard.benchmark import (
    BenchmarkComparison,
    BenchmarkDataset,
    evaluate_all_languages,
    evaluate_benchmark,
    load_benchmark,
)
from dashboard.components import (
    inject_dashboard_styles,
    render_benchmark_comparison,
    render_benchmark_example,
    render_evaluation_details,
    render_hero,
    render_metric_cards,
)
from dashboard.utils import (
    EvaluationReport,
    InputValidationError,
    samples_from_rows,
    run_evaluation,
)
from dashboard.visualizations import (
    benchmark_metric_figure,
    language_comparison_figure,
    metric_comparison_figure,
    radar_figure,
    score_distribution_figure,
)


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
BENCHMARK_PATH = ROOT_DIR / "data" / "benchmark.json"
LANGUAGE_LABELS = {
    "hindi": "Hindi",
    "marathi": "Marathi",
    "tamil": "Tamil",
    "bengali": "Bengali",
    "telugu": "Telugu",
    "gujarati": "Gujarati",
    "english": "English",
}
EMPTY_BATCH_ROWS = [
    {"question": "", "contexts": "", "answer": ""},
    {"question": "", "contexts": "", "answer": ""},
]


def main() -> None:
    """Configure and render the BharatRAG dashboard."""
    st.set_page_config(
        page_title="BharatRAG Evaluation Studio",
        page_icon="🇮🇳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _initialise_state()
    inject_dashboard_styles()
    dataset = _load_dataset()

    language, mode = _render_sidebar(dataset)
    render_hero()
    evaluation_tab, analytics_tab, explorer_tab, comparison_tab = st.tabs(
        ["Evaluation", "Analytics", "Benchmark Explorer", "Language Comparison"]
    )

    with evaluation_tab:
        if mode == "Manual Input":
            _render_manual_evaluation(language)
        else:
            _render_benchmark_evaluation(dataset, language)

    with analytics_tab:
        _render_analytics()

    with explorer_tab:
        _render_benchmark_explorer(dataset, language)

    with comparison_tab:
        _render_language_comparison(dataset)


def _initialise_state() -> None:
    """Create session-state keys used to preserve results across tab changes."""
    defaults = {
        "evaluation_report": None,
        "benchmark_comparison": None,
        "language_comparisons": None,
        "active_result_kind": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _load_dataset() -> Optional[BenchmarkDataset]:
    """Load the local benchmark without blocking manual evaluations on failure."""
    try:
        return load_benchmark(str(BENCHMARK_PATH))
    except (FileNotFoundError, ValueError) as error:
        st.error("The benchmark explorer is unavailable: {error}".format(error=error))
        LOGGER.exception("Unable to load dashboard benchmark")
        return None


def _render_sidebar(dataset: Optional[BenchmarkDataset]) -> tuple[str, str]:
    """Render global language and evaluation mode controls."""
    with st.sidebar:
        st.markdown("## BharatRAG")
        st.caption("Evaluation Studio")
        language = st.selectbox(
            "Evaluation language",
            options=list(LANGUAGE_LABELS),
            format_func=lambda value: LANGUAGE_LABELS[value],
            help="Select the language used by BharatRAG's embedding model.",
        )
        mode = st.radio(
            "Evaluation mode",
            options=("Manual Input", "Benchmark Dataset"),
            help="Enter your own RAG outputs or compare benchmark answer variants.",
        )
        st.divider()
        st.caption("Metric thresholds")
        st.caption("Poor < 0.40 · Moderate < 0.70 · Good ≥ 0.70")
        if dataset is not None:
            st.caption(
                "Bundled benchmark: {count} examples · {languages}".format(
                    count=len(dataset.examples),
                    languages=", ".join(language.capitalize() for language in dataset.languages),
                )
            )
    return language, mode


def _render_manual_evaluation(language: str) -> None:
    """Render single and multi-sample manual evaluation workflows."""
    st.subheader("Evaluate your RAG output")
    st.caption(
        "Provide a question, the retrieved context chunks, and the generated answer. "
        "No answer data leaves this local Streamlit process."
    )
    input_kind = st.radio(
        "Input shape", ("Single Example", "Multiple Examples"), horizontal=True
    )

    if input_kind == "Single Example":
        _render_single_example_form(language)
    else:
        _render_multiple_example_form(language)

    report = st.session_state.get("evaluation_report")
    if isinstance(report, EvaluationReport):
        st.divider()
        st.subheader("Latest Manual Evaluation")
        st.caption(
            "{count} sample(s) · {language}".format(
                count=len(report.samples), language=report.language.capitalize()
            )
        )
        render_metric_cards(report.summary)
        st.divider()
        render_evaluation_details(report)


def _render_single_example_form(language: str) -> None:
    """Render the manual single-example inputs and action."""
    question = st.text_area(
        "Question",
        key="single_question",
        placeholder="What did the user ask?",
        height=100,
    )
    contexts = st.text_area(
        "Retrieved Context",
        key="single_contexts",
        placeholder="Paste one retrieved chunk per line.",
        height=150,
        help="Each non-empty line is evaluated as a separate context chunk.",
    )
    answer = st.text_area(
        "Generated Answer",
        key="single_answer",
        placeholder="Paste the answer produced by your RAG system.",
        height=120,
    )
    if st.button(
        "Run BharatRAG Evaluation",
        type="primary",
        use_container_width=True,
        key="run_single_evaluation",
    ):
        _run_manual_samples(
            rows=[{"question": question, "contexts": contexts, "answer": answer}],
            language=language,
        )


def _render_multiple_example_form(language: str) -> None:
    """Render a dynamic table for batch evaluation input."""
    st.caption("Use one row per example and one context chunk per line in the context column.")
    rows = st.data_editor(
        EMPTY_BATCH_ROWS,
        key="batch_samples",
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "question": st.column_config.TextColumn("Question", required=True, width="large"),
            "contexts": st.column_config.TextColumn(
                "Retrieved Contexts", required=True, width="large"
            ),
            "answer": st.column_config.TextColumn("Generated Answer", required=True, width="large"),
        },
    )
    if st.button(
        "Run BharatRAG Evaluation",
        type="primary",
        use_container_width=True,
        key="run_batch_evaluation",
    ):
        _run_manual_samples(rows=rows, language=language)


def _run_manual_samples(rows: Sequence[Dict[str, object]], language: str) -> None:
    """Validate manual rows, run BharatRAG, and persist the report."""
    try:
        samples = samples_from_rows(rows)
        with st.spinner(
            "Loading the {language} model and evaluating your RAG output...".format(
                language=LANGUAGE_LABELS[language]
            )
        ):
            report = run_evaluation(samples, language)
    except (InputValidationError, TypeError, ValueError) as error:
        st.error("Evaluation could not start: {error}".format(error=error))
        return
    except Exception as error:  # Model loading can raise environment-specific errors.
        LOGGER.exception("Manual dashboard evaluation failed")
        st.error(
            "BharatRAG could not complete the evaluation. Check model access and "
            "try again. Details: {error}".format(error=error)
        )
        return

    st.session_state.evaluation_report = report
    st.session_state.active_result_kind = "manual"
    st.success("Evaluation complete for {count} sample(s).".format(count=len(report.samples)))


def _render_benchmark_evaluation(
    dataset: Optional[BenchmarkDataset], language: str
) -> None:
    """Render benchmark execution controls and the correct-vs-hallucinated table."""
    st.subheader("Benchmark Dataset Evaluation")
    if dataset is None:
        st.info("Add a valid data/benchmark.json file to use benchmark evaluation.")
        return
    examples = dataset.for_language(language)
    if not examples:
        st.info(
            "The bundled benchmark has no {language} examples. Choose one of: {available}.".format(
                language=LANGUAGE_LABELS[language],
                available=", ".join(item.capitalize() for item in dataset.languages),
            )
        )
        return

    st.caption(
        "Evaluate {count} {language} benchmark examples twice: first with their "
        "correct answer, then with their deliberately hallucinated answer.".format(
            count=len(examples), language=language.capitalize()
        )
    )
    if st.button(
        "Run Benchmark Evaluation",
        type="primary",
        use_container_width=True,
        key="run_benchmark_evaluation",
    ):
        try:
            with st.spinner(
                "Evaluating correct and hallucinated {language} benchmark answers...".format(
                    language=language.capitalize()
                )
            ):
                comparison = evaluate_benchmark(dataset, language)
        except Exception as error:  # Model loading and inference errors are surfaced to the user.
            LOGGER.exception("Benchmark dashboard evaluation failed")
            st.error(
                "Benchmark evaluation could not complete. Check model access and try "
                "again. Details: {error}".format(error=error)
            )
        else:
            st.session_state.benchmark_comparison = comparison
            st.session_state.active_result_kind = "benchmark"
            st.success("Benchmark evaluation complete.")

    comparison = st.session_state.get("benchmark_comparison")
    if isinstance(comparison, BenchmarkComparison) and comparison.language == language:
        st.divider()
        render_benchmark_comparison(comparison)


def _render_analytics() -> None:
    """Render visual analysis for the most recently run evaluation."""
    st.subheader("Visual Analytics")
    active_kind = st.session_state.get("active_result_kind")
    if active_kind == "manual":
        report = st.session_state.get("evaluation_report")
        if isinstance(report, EvaluationReport):
            st.plotly_chart(
                metric_comparison_figure(
                    report.summary,
                    "Metric Comparison — {language}".format(
                        language=report.language.capitalize()
                    ),
                ),
                use_container_width=True,
            )
            distribution = score_distribution_figure(report)
            if distribution is None:
                st.info("Score distribution becomes available after evaluating two or more samples.")
            else:
                st.plotly_chart(distribution, use_container_width=True)
            st.plotly_chart(
                radar_figure({"Current Evaluation": report.summary}),
                use_container_width=True,
            )
            return

    if active_kind == "benchmark":
        comparison = st.session_state.get("benchmark_comparison")
        if isinstance(comparison, BenchmarkComparison):
            st.plotly_chart(benchmark_metric_figure(comparison), use_container_width=True)
            st.plotly_chart(
                radar_figure(
                    {
                        "Correct Answer": comparison.correct,
                        "Hallucinated Answer": comparison.hallucinated,
                    }
                ),
                use_container_width=True,
            )
            st.info("Score distribution is available for multi-sample manual evaluations.")
            return

    st.info("Run a manual or benchmark evaluation to unlock charts and score comparisons.")


def _render_benchmark_explorer(
    dataset: Optional[BenchmarkDataset], selected_language: str
) -> None:
    """Render a filtered, paginated browser for benchmark records."""
    st.subheader("Benchmark Explorer")
    if dataset is None:
        st.info("A valid benchmark file is required to browse samples.")
        return

    default_index = dataset.languages.index(selected_language) if selected_language in dataset.languages else 0
    language = st.selectbox(
        "Language Filter",
        options=list(dataset.languages),
        index=default_index,
        format_func=lambda value: LANGUAGE_LABELS.get(value, value.capitalize()),
        key="explorer_language",
    )
    examples = dataset.for_language(language)
    categories = dataset.categories_for_language(language)
    overview = st.columns(3)
    overview[0].metric("Examples", len(examples))
    overview[1].metric("Language", language.capitalize())
    overview[2].metric("Categories", len(categories))
    st.caption("Categories: {categories}".format(
        categories=", ".join(category.replace("_", " ").title() for category in categories)
    ))

    index_key = "benchmark_position_{language}".format(language=language)
    position = int(st.session_state.get(index_key, 0))
    position = min(max(position, 0), len(examples) - 1)
    navigation = st.columns((1, 1, 6))
    if navigation[0].button("Previous", disabled=position == 0, key="previous_{language}".format(language=language)):
        position -= 1
        st.session_state[index_key] = position
    if navigation[1].button("Next", disabled=position >= len(examples) - 1, key="next_{language}".format(language=language)):
        position += 1
        st.session_state[index_key] = position
    render_benchmark_example(examples[position], position + 1, len(examples))


def _render_language_comparison(dataset: Optional[BenchmarkDataset]) -> None:
    """Render the opt-in all-language benchmark comparison view."""
    st.subheader("Language Comparison")
    st.caption(
        "Compare average benchmark results across languages. This first run loads "
        "each language model and may take a few minutes. Results remain available "
        "for the current browser session."
    )
    if dataset is None:
        st.info("A valid benchmark file is required for language comparison.")
        return

    if st.button(
        "Compare Benchmark Languages",
        type="primary",
        key="compare_benchmark_languages",
    ):
        try:
            with st.spinner("Evaluating all benchmark languages. This can take a few minutes..."):
                comparisons = evaluate_all_languages(dataset)
        except Exception as error:  # Different language models can fail independently.
            LOGGER.exception("Language comparison failed")
            st.error(
                "Language comparison could not complete. Check model access and try "
                "again. Details: {error}".format(error=error)
            )
        else:
            st.session_state.language_comparisons = comparisons
            st.success("Language comparison complete.")

    comparisons = st.session_state.get("language_comparisons")
    if not isinstance(comparisons, dict) or not comparisons:
        st.info("Select “Compare Benchmark Languages” to calculate average scores.")
        return

    st.plotly_chart(language_comparison_figure(comparisons), use_container_width=True)
    rows = []
    for language, comparison in comparisons.items():
        rows.append(
            {
                "Language": language.capitalize(),
                "Correct Overall": comparison.correct.overall,
                "Hallucinated Overall": comparison.hallucinated.overall,
                "Correct Groundedness": comparison.correct.groundedness,
                "Hallucinated Groundedness": comparison.hallucinated.groundedness,
            }
        )
    st.dataframe(
        rows,
        hide_index=True,
        use_container_width=True,
        column_config={
            column: st.column_config.NumberColumn(format="%.3f")
            for column in (
                "Correct Overall",
                "Hallucinated Overall",
                "Correct Groundedness",
                "Hallucinated Groundedness",
            )
        },
    )


if __name__ == "__main__":
    main()

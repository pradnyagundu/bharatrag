"""Reusable Streamlit components for the BharatRAG dashboard."""

from __future__ import annotations

from html import escape
import streamlit as st

from dashboard.benchmark import BenchmarkComparison, BenchmarkExample, comparison_rows
from dashboard.theme import dashboard_css
from dashboard.utils import EvaluationReport, METRIC_LABELS, MetricScores, score_status


def inject_dashboard_styles() -> None:
    """Inject the shared token-based visual system."""
    st.markdown(dashboard_css(), unsafe_allow_html=True)


def render_hero() -> None:
    """Render the dashboard's introductory banner."""
    st.markdown(
        """
        <section class="br-hero">
            <div class="br-eyebrow">BharatRAG Evaluation Studio</div>
            <h1>Measure RAG quality across Indian languages.</h1>
            <p>Evaluate retrieval relevance, answer groundedness, and answer relevance with BharatRAG's offline metric suite.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(scores: MetricScores) -> None:
    """Render four status-aware metric cards with visual progress bars."""
    columns = st.columns(4)
    for column, metric in zip(
        columns,
        ("overall", "context_relevance", "groundedness", "answer_relevance"),
    ):
        with column:
            _render_metric_card(METRIC_LABELS[metric], scores.as_dict()[metric])


def render_evaluation_details(report: EvaluationReport) -> None:
    """Render a disclosure-friendly per-sample evaluation section."""
    st.subheader("Evaluation Details")
    st.caption(
        "Per-sample scores use the same BharatRAG metric implementations as the aggregate result."
    )
    for index, evaluation in enumerate(report.samples, start=1):
        with st.expander("Sample {index} · Overall {score:.2f}".format(
            index=index, score=evaluation.scores.overall
        )):
            st.caption("Question")
            st.write(evaluation.sample.question)
            st.caption("Retrieved Context")
            for context_index, context in enumerate(evaluation.sample.contexts, start=1):
                st.markdown("**Chunk {index}**".format(index=context_index))
                st.write(context)
            st.caption("Generated Answer")
            st.write(evaluation.sample.answer)

            st.caption("Metric Breakdown")
            metrics = evaluation.scores.as_dict()
            metric_columns = st.columns(4)
            for column, metric in zip(
                metric_columns,
                ("context_relevance", "groundedness", "answer_relevance", "overall"),
            ):
                with column:
                    value = metrics[metric]
                    st.metric(METRIC_LABELS[metric], "{:.2f}".format(value))
                    st.progress(min(max(value, 0.0), 1.0))

            if evaluation.scores.groundedness < 0.5:
                st.warning(
                    "Potential Hallucination Detected — groundedness is below 0.50."
                )


def render_benchmark_comparison(comparison: BenchmarkComparison) -> None:
    """Render benchmark result cards and the requested answer-type table."""
    st.subheader("Benchmark Evaluation Results")
    st.caption(
        "{count} {language} examples evaluated against correct and deliberately "
        "hallucinated answers.".format(
            count=comparison.sample_count, language=comparison.language.capitalize()
        )
    )
    render_metric_cards(comparison.correct)
    st.caption("Metric cards show the correct-answer run.")
    st.dataframe(
        comparison_rows(comparison),
        hide_index=True,
        use_container_width=True,
        column_config={
            "Overall": st.column_config.NumberColumn(format="%.3f"),
            "Context": st.column_config.NumberColumn(format="%.3f"),
            "Groundedness": st.column_config.NumberColumn(format="%.3f"),
            "Answer Relevance": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    delta = comparison.correct.overall - comparison.hallucinated.overall
    if delta > 0:
        st.success(
            "Correct answers lead hallucinated answers by {delta:.3f} overall.".format(
                delta=delta
            )
        )
    else:
        st.info(
            "This subset did not produce a higher correct-answer overall score. "
            "Inspect the detailed metric table before drawing conclusions."
        )


def render_benchmark_example(example: BenchmarkExample, position: int, total: int) -> None:
    """Render the currently selected benchmark sample."""
    st.subheader("Sample {position} of {total}".format(position=position, total=total))
    overview = st.columns(3)
    overview[0].metric("Examples", total)
    overview[1].metric("Language", example.language.capitalize())
    overview[2].metric("Category", example.category.replace("_", " ").title())

    st.caption("Question")
    st.write(example.question)
    st.caption("Retrieved Context")
    for index, context in enumerate(example.contexts, start=1):
        st.markdown("**Chunk {index}**".format(index=index))
        st.write(context)
    answer_columns = st.columns(2)
    with answer_columns[0]:
        st.caption("Correct Answer")
        st.success(example.correct_answer)
    with answer_columns[1]:
        st.caption("Hallucinated Answer")
        st.error(example.hallucinated_answer)


def _render_metric_card(label: str, score: float) -> None:
    status = score_status(score)
    status_class = status.lower()
    score = min(max(score, 0.0), 1.0)
    st.markdown(
        """
        <section class="br-card">
            <div class="br-card-label">{label}</div>
            <div class="br-score-row">
                <div class="br-score">{score:.2f}</div>
                <span class="br-status br-status--{status_class}">{status}</span>
            </div>
            <div class="br-progress"><div class="br-progress-fill br-progress-fill--{status_class}" style="width:{progress:.1f}%;"></div></div>
            <div class="br-caption">{progress:.0f}% quality signal</div>
        </section>
        """.format(
            label=escape(label),
            score=score,
            status=status,
            status_class=status_class,
            progress=score * 100,
        ),
        unsafe_allow_html=True,
    )

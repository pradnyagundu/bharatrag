"""Plotly figure factories used by the BharatRAG dashboard."""

from __future__ import annotations

from typing import Mapping, Optional

import plotly.graph_objects as go

from dashboard.benchmark import BenchmarkComparison
from dashboard.theme import (
    ACCENT,
    BORDER,
    DANGER,
    PRIMARY,
    SUCCESS,
    SURFACE,
    TEXT_PRIMARY,
    WARNING,
)
from dashboard.utils import METRIC_LABELS, EvaluationReport, MetricScores


_METRIC_ORDER = (
    "context_relevance",
    "groundedness",
    "answer_relevance",
    "overall",
)
_RADAR_METRICS = _METRIC_ORDER[:3]
def metric_comparison_figure(scores: MetricScores, title: str) -> go.Figure:
    """Build a bar chart for a single evaluation's four metrics."""
    figure = go.Figure(
        go.Bar(
            x=[METRIC_LABELS[metric] for metric in _METRIC_ORDER],
            y=[scores.as_dict()[metric] for metric in _METRIC_ORDER],
            marker_color=[PRIMARY, ACCENT, PRIMARY, WARNING],
            text=["{:.2f}".format(scores.as_dict()[metric]) for metric in _METRIC_ORDER],
            textposition="outside",
            hovertemplate="%{x}<br>Score: %{y:.3f}<extra></extra>",
        )
    )
    return _style_figure(figure, title)


def score_distribution_figure(report: EvaluationReport) -> Optional[go.Figure]:
    """Build the overall-score histogram when a run has multiple samples."""
    if len(report.samples) < 2:
        return None

    figure = go.Figure(
        go.Histogram(
            x=[sample.scores.overall for sample in report.samples],
            xbins={"start": 0, "end": 1, "size": 0.1},
            marker={"color": PRIMARY, "line": {"color": SURFACE, "width": 1}},
            hovertemplate="Score range: %{x}<br>Samples: %{y}<extra></extra>",
        )
    )
    figure.update_layout(xaxis_title="Overall score", yaxis_title="Sample count")
    figure.update_xaxes(range=[0, 1], dtick=0.2)
    return _style_figure(figure, "Overall Score Distribution")


def radar_figure(series: Mapping[str, MetricScores]) -> go.Figure:
    """Build a radar chart for the three RAG-triad metrics."""
    figure = go.Figure()
    palette = (PRIMARY, ACCENT, SUCCESS, DANGER)
    categories = [METRIC_LABELS[metric] for metric in _RADAR_METRICS]
    for index, (label, scores) in enumerate(series.items()):
        values = [scores.as_dict()[metric] for metric in _RADAR_METRICS]
        figure.add_trace(
            go.Scatterpolar(
                r=values + values[:1],
                theta=categories + categories[:1],
                fill="toself",
                fillcolor=_with_alpha(palette[index % len(palette)], 0.15),
                line={"color": palette[index % len(palette)], "width": 2},
                name=label,
                hovertemplate="%{theta}<br>Score: %{r:.3f}<extra>" + label + "</extra>",
            )
        )
    figure.update_layout(
        title="RAG Triad Profile",
        polar={
            "radialaxis": {"visible": True, "range": [0, 1], "gridcolor": BORDER},
            "angularaxis": {"gridcolor": BORDER},
            "bgcolor": SURFACE,
        },
        height=390,
        margin={"l": 44, "r": 44, "t": 62, "b": 30},
        paper_bgcolor=SURFACE,
        font={"color": TEXT_PRIMARY},
        legend={"orientation": "h", "y": -0.15},
    )
    return figure


def benchmark_metric_figure(comparison: BenchmarkComparison) -> go.Figure:
    """Compare correct and hallucinated answer variants across all metrics."""
    figure = go.Figure()
    labels = [METRIC_LABELS[metric] for metric in _METRIC_ORDER]
    figure.add_trace(
        go.Bar(
            name="Correct Answer",
            x=labels,
            y=[comparison.correct.as_dict()[metric] for metric in _METRIC_ORDER],
            marker_color=SUCCESS,
            hovertemplate="%{x}<br>Correct: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Hallucinated Answer",
            x=labels,
            y=[comparison.hallucinated.as_dict()[metric] for metric in _METRIC_ORDER],
            marker_color=DANGER,
            hovertemplate="%{x}<br>Hallucinated: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(barmode="group")
    return _style_figure(
        figure,
        "Correct vs. Hallucinated Answers — {language}".format(
            language=comparison.language.capitalize()
        ),
    )


def language_comparison_figure(
    comparisons: Mapping[str, BenchmarkComparison]
) -> go.Figure:
    """Build a grouped language comparison using average overall scores."""
    languages = [language.capitalize() for language in comparisons]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            name="Correct Answer",
            x=languages,
            y=[comparison.correct.overall for comparison in comparisons.values()],
            marker_color=SUCCESS,
            hovertemplate="%{x}<br>Correct overall: %{y:.3f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            name="Hallucinated Answer",
            x=languages,
            y=[comparison.hallucinated.overall for comparison in comparisons.values()],
            marker_color=DANGER,
            hovertemplate="%{x}<br>Hallucinated overall: %{y:.3f}<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        xaxis_title="Benchmark language",
        yaxis_title="Average overall score",
    )
    figure.update_yaxes(range=[0, 1])
    return _style_figure(figure, "Average Overall Score by Language")


def _style_figure(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title={"text": title, "font": {"size": 18, "color": TEXT_PRIMARY}},
        height=380,
        margin={"l": 20, "r": 20, "t": 64, "b": 24},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": TEXT_PRIMARY},
        legend={"orientation": "h", "y": -0.18},
    )
    figure.update_yaxes(range=[0, 1.08], gridcolor=BORDER, zeroline=False)
    figure.update_xaxes(showgrid=False)
    return figure


def _with_alpha(hex_color: str, alpha: float) -> str:
    """Convert a six-character hex colour to an rgba Plotly colour."""
    red = int(hex_color[1:3], 16)
    green = int(hex_color[3:5], 16)
    blue = int(hex_color[5:7], 16)
    return "rgba({red}, {green}, {blue}, {alpha})".format(
        red=red, green=green, blue=blue, alpha=alpha
    )

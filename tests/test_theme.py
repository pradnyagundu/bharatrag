"""Tests for the dashboard's shared visual design tokens."""

from dashboard.theme import (
    ACCENT,
    BACKGROUND,
    BORDER,
    DANGER,
    PRIMARY,
    PRIMARY_HOVER,
    SIDEBAR,
    SUCCESS,
    SURFACE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    WARNING,
    dashboard_css,
)


def test_theme_uses_the_dashboard_palette():
    assert {
        PRIMARY,
        PRIMARY_HOVER,
        ACCENT,
        SUCCESS,
        WARNING,
        DANGER,
        BACKGROUND,
        SURFACE,
        SIDEBAR,
        TEXT_PRIMARY,
        TEXT_SECONDARY,
        BORDER,
    } == {
        "#101330",
        "#0A1255",
        "#14B8A6",
        "#10B981",
        "#F59E0B",
        "#EF4444",
        "#E8EAEF",
        "#F9F8F8",
        "#0A1225",
        "#0F172A",
        "#64748B",
        "#1A1C21",
    }


def test_dashboard_css_contains_visible_keyboard_focus_treatment():
    css = dashboard_css()

    assert "focus-visible" in css
    assert "--br-primary" in css
    assert "stTextArea" in css

"""Shared visual tokens for BharatRAG's dashboard and Plotly figures.

Keep every application-owned colour in this module. Streamlit's native theme
uses the matching values in ``.streamlit/config.toml``.
"""

from __future__ import annotations

PRIMARY = "#101330"
PRIMARY_HOVER = "#0A1255"
ACCENT = "#14B8A6"
SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
BACKGROUND = "#E8EAEF"
SURFACE = "#F9F8F8"
SIDEBAR = "#0A1225"
TEXT_PRIMARY = "#0F172A"
TEXT_SECONDARY = "#64748B"
BORDER = "#1A1C21"

PRIMARY_SOFT = "rgba(79, 70, 229, 0.10)"
ACCENT_SOFT = "rgba(20, 184, 166, 0.12)"
SUCCESS_SOFT = "rgba(16, 185, 129, 0.12)"
WARNING_SOFT = "rgba(245, 158, 11, 0.14)"
DANGER_SOFT = "rgba(239, 68, 68, 0.12)"
SIDEBAR_MUTED = "rgba(226, 232, 240, 0.72)"
SIDEBAR_HOVER = "rgba(255, 255, 255, 0.08)"
FOCUS_RING = "rgba(79, 70, 229, 0.22)"
SHADOW = "rgba(15, 23, 42, 0.08)"

def dashboard_css() -> str:
    """Return the dashboard stylesheet built from the shared token set.

    Streamlit does not offer an API for styling every native widget. The rules
    below only adjust presentation; widget semantics and interaction behaviour
    remain Streamlit-owned.
    """
    return """
    <style>
        :root {{
            --br-primary: {primary};
            --br-primary-hover: {primary_hover};
            --br-accent: {accent};
            --br-success: {success};
            --br-warning: {warning};
            --br-danger: {danger};
            --br-background: {background};
            --br-surface: {surface};
            --br-sidebar: {sidebar};
            --br-text: {text_primary};
            --br-text-secondary: {text_secondary};
            --br-border: {border};
            --br-primary-soft: {primary_soft};
            --br-accent-soft: {accent_soft};
            --br-success-soft: {success_soft};
            --br-warning-soft: {warning_soft};
            --br-danger-soft: {danger_soft};
            --br-sidebar-muted: {sidebar_muted};
            --br-sidebar-hover: {sidebar_hover};
            --br-focus-ring: {focus_ring};
            --br-shadow: {shadow};
        }}

        .stApp {{
            background: var(--br-background);
            color: var(--br-text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
                "Segoe UI", sans-serif;
        }}

        .block-container {{
            margin-top: 1.5rem;
            max-width: 1440px;
            padding: 1.5rem 2rem 3rem;
        }}

        h1, h2, h3 {{
            color: var(--br-text);
            font-weight: 650;
            letter-spacing: -0.055em;
        }}

        h2 {{
            font-size: 1.35rem;
            margin-bottom: 0.25rem;
        }}

        [data-testid="stCaptionContainer"] {{
            color: var(--br-text-secondary);
            line-height: 1.5;
        }}

        [data-testid="stSidebar"] {{
            background: var(--br-sidebar);
            border-right: 1px solid var(--br-sidebar-hover);
        }}

        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
            padding: 1.25rem 0.9rem 1.5rem;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
            color: var(--br-surface);
        }}

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
            color: var(--br-sidebar-muted);
        }}

        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
            color: var(--br-surface);
            font-size: 0.8rem;
            font-weight: 650;
            letter-spacing: 0.01em;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] > div {{
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-radius: 9px;
            box-shadow: none;
        }}

        [data-testid="stSidebar"] [data-baseweb="select"] input,
        [data-testid="stSidebar"] [data-baseweb="select"] svg,
        [data-testid="stSidebar"] [data-baseweb="select"] div {{
            color: var(--br-text);
        }}

        [data-testid="stSidebar"] .stRadio > div {{
            gap: 0.45rem;
        }}

        [data-testid="stSidebar"] .stRadio label {{
            align-items: center;
            border-radius: 8px;
            color: var(--br-sidebar-muted);
            margin: 0;
            min-height: 2.25rem;
            padding: 0.25rem 0.45rem;
            transition: background-color 150ms ease, color 150ms ease;
        }}

        [data-testid="stSidebar"] .stRadio label:hover {{
            background: var(--br-sidebar-hover);
            color: var(--br-surface);
        }}

        [data-testid="stSidebar"] [data-testid="stDivider"] {{
            border-color: var(--br-sidebar-hover);
            margin: 1.25rem 0;
        }}

        .br-hero {{
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-left: 4px solid var(--br-primary);
            border-radius: 14px;
            box-shadow: 0 1px 2px var(--br-shadow);
            margin: 0 0 1.4rem;
            padding: 1.35rem 1.5rem;
        }}

        .br-eyebrow {{
            color: var(--br-primary);
            font-size: 01rem;
            font-weight: 700;
            letter-spacing: 0.45 em;
            text-transform: uppercase;
        }}

        .br-hero h1 {{
            color: var(--br-text);
            font-size: clamp(1.65rem, 2.6vw, 2.25rem);
            font-weight: 680;
            letter-spacing: -0.045em;
            line-height: 1.16;
            margin: 0.35rem 0 0.45rem;
        }}

        .br-hero p {{
            color: var(--br-text-secondary);
            font-size: 0.97rem;
            line-height: 1.55;
            margin: 0;
            max-width: 52rem;
        }}

        .br-card {{
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-radius: 12px;
            box-shadow: 0 1px 2px var(--br-shadow);
            min-height: 148px;
            padding: 1rem 1rem 1.05rem;
        }}

        .br-card-label {{
            color: var(--br-text-secondary);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        .br-score-row {{
            align-items: center;
            display: flex;
            gap: 0.6rem;
            justify-content: space-between;
            margin: 0.7rem 0;
        }}

        .br-score {{
            color: var(--br-text);
            font-size: 1.9rem;
            font-weight: 700;
            letter-spacing: -0.05em;
            line-height: 1;
        }}

        .br-status {{
            align-items: center;
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-radius: 999px;
            color: var(--br-text);
            display: inline-flex;
            font-size: 0.72rem;
            font-weight: 700;
            gap: 0.35rem;
            padding: 0.26rem 0.55rem;
        }}

        .br-status::before {{
            background: currentColor;
            border-radius: 50%;
            content: "";
            display: inline-block;
            height: 0.45rem;
            width: 0.45rem;
        }}

        .br-status--good {{ background: var(--br-success-soft); border-color: var(--br-success); }}
        .br-status--moderate {{ background: var(--br-warning-soft); border-color: var(--br-warning); }}
        .br-status--poor {{ background: var(--br-danger-soft); border-color: var(--br-danger); }}
        .br-status--good::before {{ background: var(--br-success); }}
        .br-status--moderate::before {{ background: var(--br-warning); }}
        .br-status--poor::before {{ background: var(--br-danger); }}

        .br-progress {{
            background: var(--br-border);
            border-radius: 999px;
            height: 6px;
            overflow: hidden;
        }}

        .br-progress-fill {{ border-radius: inherit; height: 100%; }}
        .br-progress-fill--good {{ background: var(--br-success); }}
        .br-progress-fill--moderate {{ background: var(--br-warning); }}
        .br-progress-fill--poor {{ background: var(--br-danger); }}

        .br-caption {{
            color: var(--br-text-secondary);
            font-size: 0.75rem;
            margin-top: 0.65rem;
        }}

        [data-testid="stTextArea"] textarea,
        [data-testid="stTextInput"] input,
        [data-testid="stDataEditor"] input,
        [data-testid="stDataEditor"] textarea {{
            background: var(--br-surface) !important;
            border: 1px solid var(--br-border) !important;
            border-radius: 9px !important;
            color: var(--br-text) !important;
            font-size: 0.92rem !important;
        }}

        [data-testid="stTextArea"] textarea {{
            line-height: 1.5;
            padding: 0.7rem 0.8rem;
        }}

        [data-testid="stTextArea"] textarea::placeholder,
        [data-testid="stTextInput"] input::placeholder {{
            color: var(--br-text-secondary);
            opacity: 0.72;
        }}

        [data-testid="stTextArea"] textarea:focus,
        [data-testid="stTextInput"] input:focus,
        [data-testid="stDataEditor"] input:focus,
        [data-testid="stDataEditor"] textarea:focus {{
            border-color: var(--br-primary) !important;
            box-shadow: 0 0 0 3px var(--br-focus-ring) !important;
            outline: none !important;
        }}

        [data-testid="stDataEditor"] {{
            border: 1px solid var(--br-border);
            border-radius: 10px;
            overflow: hidden;
        }}

        .stButton > button {{
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-radius: 9px;
            box-shadow: none;
            color: var(--br-text);
            font-weight: 650;
            min-height: 2.4rem;
            padding: 0.45rem 0.9rem;
            transition: background-color 150ms ease, border-color 150ms ease,
                box-shadow 150ms ease, transform 150ms ease;
        }}

        .stButton > button[kind="primary"] {{
            background: var(--br-primary);
            border-color: var(--br-primary);
            box-shadow: 0 3px 8px var(--br-focus-ring);
            color: var(--br-surface);
        }}

        .stButton > button[kind="primary"]:hover {{
            background: var(--br-primary-hover);
            border-color: var(--br-primary-hover);
            box-shadow: 0 5px 14px var(--br-focus-ring);
            color: var(--br-surface);
            transform: translateY(-1px);
        }}

        .stButton > button:not([kind="primary"]):hover {{
            background: var(--br-primary-soft);
            border-color: var(--br-primary);
            color: var(--br-primary);
        }}

        .stButton > button:disabled {{
            background: var(--br-background);
            border-color: var(--br-border);
            box-shadow: none;
            color: var(--br-text-secondary);
            opacity: 0.72;
            transform: none;
        }}

        :where(button, input, textarea, [role="tab"]):focus-visible {{
            outline: 2px solid var(--br-primary) !important;
            outline-offset: 2px;
        }}

        [data-testid="stTabs"] [data-baseweb="tab-list"] {{
            border-bottom: 1px solid var(--br-border);
            gap: 0.25rem;
            margin-bottom: 1.25rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px 8px 0 0;
            color: var(--br-text-secondary);
            font-size: 0.9rem;
            font-weight: 650;
            margin-bottom: -1px;
            padding: 0.6rem 0.85rem;
        }}

        [data-testid="stTabs"] [data-baseweb="tab"]:hover {{
            background: var(--br-primary-soft);
            color: var(--br-primary);
        }}

        [data-testid="stTabs"] [aria-selected="true"] {{
            background: var(--br-primary-soft);
            color: var(--br-primary);
        }}

        [data-testid="stExpander"] {{
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-radius: 10px;
            box-shadow: 0 1px 2px var(--br-shadow);
        }}

        [data-testid="stExpander"]:hover {{ border-color: var(--br-primary); }}

        [data-testid="stMetric"] {{
            background: var(--br-surface);
            border: 1px solid var(--br-border);
            border-radius: 10px;
            padding: 0.75rem;
        }}

        [data-testid="stMetricLabel"] {{ color: var(--br-text-secondary); }}
        [data-testid="stMetricValue"] {{ color: var(--br-text); }}

        [data-testid="stProgressBar"] > div > div {{ background: var(--br-primary); }}

        [data-testid="stDataFrame"] {{
            border: 1px solid var(--br-border);
            border-radius: 10px;
            overflow: hidden;
        }}

        [data-testid="stHorizontalBlock"] {{ gap: 1rem; }}

        @media (max-width: 900px) {{
            .block-container {{ padding: 1.1rem 1rem 2rem; }}
            .br-hero {{ padding: 1.15rem 1.2rem; }}
            [data-testid="stTabs"] [data-baseweb="tab"] {{ padding: 0.55rem 0.65rem; }}
        }}
    </style>
    """.format(
        primary=PRIMARY,
        primary_hover=PRIMARY_HOVER,
        accent=ACCENT,
        success=SUCCESS,
        warning=WARNING,
        danger=DANGER,
        background=BACKGROUND,
        surface=SURFACE,
        sidebar=SIDEBAR,
        text_primary=TEXT_PRIMARY,
        text_secondary=TEXT_SECONDARY,
        border=BORDER,
        primary_soft=PRIMARY_SOFT,
        accent_soft=ACCENT_SOFT,
        success_soft=SUCCESS_SOFT,
        warning_soft=WARNING_SOFT,
        danger_soft=DANGER_SOFT,
        sidebar_muted=SIDEBAR_MUTED,
        sidebar_hover=SIDEBAR_HOVER,
        focus_ring=FOCUS_RING,
        shadow=SHADOW,
    )

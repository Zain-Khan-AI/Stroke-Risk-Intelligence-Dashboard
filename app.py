from pathlib import Path
from html import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st


DATA_PATH = Path("healthcare-dataset-stroke-data.csv")

PRIMARY = "#3b82f6"
PRIMARY_DARK = "#60a5fa"
ACCENT = "#2dd4bf"
ACCENT_SOFT = "#5eead4"
DANGER = "#ef4444"
WARNING = "#fbbf24"
SUCCESS = "#22c55e"
INK = "#f1f5f9"
MUTED = "#94a3b8"
PANEL = "#1e293b"
PAGE = "#0f172a"
GRID = "#334155"
NO_STROKE = "#2dd4bf"
STROKE = "#ef4444"
CHART_BG = "#1e293b"
CHART_GRID = "#334155"
GRADIENT_BASE = "linear-gradient(135deg, #0f172a 0%, #1e1b4b 32%, #0f766e 64%, #0f172a 100%)"
CLINICAL_PALETTE = ["#3b82f6", "#2dd4bf", "#fbbf24", "#ef4444", "#a78bfa", "#22d3ee"]
RISK_PALETTE = {"Low": "#22c55e", "Watch": "#fbbf24", "Elevated": "#ef4444"}

OUTCOME_PALETTE = {"No Stroke": NO_STROKE, "Stroke": STROKE}
FIG_DPI = 150
PLOTLY_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "responsive": True,
}


st.set_page_config(
    page_title="Stroke Risk Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["bmi_missing"] = data["bmi"].isna()
    data["bmi"] = data["bmi"].fillna(data["bmi"].median())
    data["stroke_label"] = data["stroke"].map({0: "No Stroke", 1: "Stroke"})
    data["hypertension_label"] = data["hypertension"].map({0: "No", 1: "Yes"})
    data["heart_disease_label"] = data["heart_disease"].map({0: "No", 1: "Yes"})
    data["age_group"] = pd.cut(
        data["age"],
        bins=[0, 18, 35, 50, 65, 120],
        labels=["0-18", "19-35", "36-50", "51-65", "66+"],
        include_lowest=True,
    )
    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Healthy", "Overweight", "Obese"],
        include_lowest=True,
    )
    data["glucose_band"] = pd.cut(
        data["avg_glucose_level"],
        bins=[0, 100, 140, 400],
        labels=["Normal", "Elevated", "High"],
        include_lowest=True,
    )
    data["risk_score"] = (
        (data["age"].clip(0, 80) / 80 * 38)
        + (data["hypertension"] * 16)
        + (data["heart_disease"] * 18)
        + (data["avg_glucose_level"].gt(140) * 12)
        + (data["avg_glucose_level"].between(100, 140, inclusive="right") * 6)
        + (data["bmi"].gt(30) * 8)
        + (data["smoking_status"].isin(["formerly smoked", "smokes"]) * 8)
    ).clip(0, 100).round(1)
    data["risk_band"] = pd.cut(
        data["risk_score"],
        bins=[-1, 24, 49, 100],
        labels=["Low", "Watch", "Elevated"],
    )
    return data


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
            :root {{
                --primary: {PRIMARY};
                --primary-dark: {PRIMARY_DARK};
                --accent: {ACCENT};
                --accent-soft: {ACCENT_SOFT};
                --danger: {DANGER};
                --warning: {WARNING};
                --success: {SUCCESS};
                --ink: {INK};
                --muted: {MUTED};
                --panel: {PANEL};
                --page: {PAGE};
                --grid: {GRID};
                --gradient-base: {GRADIENT_BASE};
            }}

            .stApp {{
                background: var(--gradient-base);
                background-attachment: fixed;
                color: var(--ink);
            }}



            [data-testid="stSidebar"] {{
                background:
                    radial-gradient(circle at 18% 8%, rgba(59,130,246,.12) 0, rgba(59,130,246,0) 38%),
                    radial-gradient(circle at 94% 28%, rgba(45,212,191,.08) 0, rgba(45,212,191,0) 34%),
                    linear-gradient(180deg, rgba(15,23,42,.98) 0%, rgba(30,27,75,.96) 40%, rgba(15,118,110,.95) 74%, rgba(15,23,42,.94) 100%);
                border-right: 1px solid rgba(59,130,246,.16);
                box-shadow: 18px 0 45px rgba(0,0,0,.08);
            }}

            [data-testid="stSidebarContent"] {{
                max-height: 100vh;
                overflow-y: auto;
                padding-bottom: 1.5rem;
                scrollbar-width: thin;
                scrollbar-color: rgba(59,130,246,.62) rgba(30,41,59,.78);
            }}

            [data-testid="stSidebarContent"]::-webkit-scrollbar {{
                width: 8px;
            }}

            [data-testid="stSidebarContent"]::-webkit-scrollbar-track {{
                background: linear-gradient(180deg, rgba(15,23,42,.94), rgba(30,27,75,.86), rgba(15,118,110,.84));
            }}

            [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {{
                background: linear-gradient(180deg, rgba(59,130,246,.86), rgba(45,212,191,.82), rgba(99,102,241,.72));
                border-radius: 999px;
                border: 2px solid rgba(255,255,255,.86);
            }}

            [data-testid="stSidebar"] * {{
                color: #cbd5e1;
            }}

            [data-testid="stSidebar"] [data-baseweb="select"] *,
            [data-testid="stSidebar"] [data-baseweb="input"] *,
            [data-testid="stSidebar"] [data-baseweb="tag"] * {{
                color: #ffffff;
            }}

            [data-testid="stSidebar"] .stButton button {{
                width: 100%;
            }}

            [data-testid="stSidebar"] h2 {{
                color: #60a5fa;
                font-size: 1.16rem;
                font-weight: 900;
                margin-bottom: .15rem;
            }}

            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] p {{
                color: #94a3b8;
                font-weight: 700;
            }}

            [data-testid="stSidebar"] [data-baseweb="select"] div,
            [data-testid="stSidebar"] [data-baseweb="tag"] {{
                color: #ffffff;
            }}

            [data-testid="stSidebar"] [data-baseweb="select"] > div,
            [data-testid="stSidebar"] [data-baseweb="input"] > div {{
                background: rgba(51,65,85,.92);
                border-color: rgba(59,130,246,.18);
                border-radius: 8px;
            }}

            [data-testid="stSidebar"] [data-baseweb="tag"] {{
                background: linear-gradient(135deg, rgba(59,130,246,.25), rgba(45,212,191,.15), rgba(99,102,241,.12));
                border: 1px solid rgba(59,130,246,.25);
                color: #e2e8f0;
            }}

            [data-testid="stSidebar"] [data-testid="stSlider"] {{
                background: rgba(51,65,85,.60);
                border: 1px solid rgba(59,130,246,.10);
                border-radius: 8px;
                padding: .55rem .65rem .2rem;
                margin-bottom: .45rem;
            }}

            [data-baseweb="popover"] ul {{
                background: #1e293b !important;
                border: 1px solid rgba(59,130,246,.2) !important;
            }}
            [data-baseweb="popover"] li {{
                color: #ffffff !important;
                background: transparent !important;
            }}
            [data-baseweb="popover"] li:hover {{
                background: rgba(59,130,246,.15) !important;
            }}

            .block-container {{
                padding-top: 1.35rem;
                padding-bottom: 2.4rem;
                max-width: 1480px;
            }}

            h1, h2, h3 {{
                letter-spacing: 0;
            }}

            .hero {{
                border: 1px solid rgba(59,130,246,.15);
                background: rgba(30,41,59,.92);
                border-radius: 8px;
                color: #f1f5f9;
                padding: 2.35rem 2.5rem 2.25rem;
                box-shadow: 0 18px 44px rgba(0,0,0,.20);
                margin-bottom: 1.35rem;
            }}

            .hero-kicker {{
                color: #f1f5f9;
                font-size: .78rem;
                font-weight: 800;
                letter-spacing: .14em;
                text-transform: uppercase;
                margin-bottom: .45rem;
            }}

            .hero h1 {{
                color: #ffffff;
                font-size: clamp(2rem, 4vw, 3.25rem);
                line-height: 1.05;
                margin: 0;
            }}

            .hero p {{
                color: #cbd5e1;
                font-size: 1.04rem;
                max-width: 780px;
                margin: .85rem 0 0;
            }}

            .kpi-heading {{
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: .25rem 0 .75rem;
            }}

            .kpi-heading-title {{
                color: var(--ink);
                font-size: 1.08rem;
                font-weight: 900;
                letter-spacing: 0;
            }}

            .kpi-heading-note {{
                color: var(--muted);
                font-size: .88rem;
                font-weight: 700;
                text-align: right;
            }}

            .kpi-spacer {{
                height: 1.35rem;
            }}

            .metric-card {{
                position: relative;
                overflow: hidden;
                background:
                    linear-gradient(180deg, rgba(30,41,59,.98), rgba(15,23,42,.96));
                border: 1px solid rgba(51,65,85,.6);
                border-radius: 8px;
                padding: 1.12rem 1.12rem 1.18rem;
                box-shadow: 0 16px 36px rgba(0,0,0,.25);
                min-height: 148px;
                margin-bottom: 1.15rem;
                transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
            }}

            .metric-card:hover {{
                transform: translateY(-3px);
                border-color: rgba(59,130,246,.4);
                box-shadow: 0 20px 44px rgba(0,0,0,.35);
            }}

            .metric-card::before {{
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 4px;
                background: linear-gradient(90deg, var(--primary), var(--accent-soft));
            }}

            .metric-card.primary::before {{
                background: linear-gradient(90deg, var(--primary), #60a5fa);
            }}

            .metric-card.accent::before {{
                background: linear-gradient(90deg, var(--accent), var(--accent-soft));
            }}

            .metric-card.warning::before {{
                background: linear-gradient(90deg, var(--warning), #fcd34d);
            }}

            .metric-card.danger::before {{
                background: linear-gradient(90deg, var(--danger), #f43f5e);
            }}

            .metric-card.success::before {{
                background: linear-gradient(90deg, var(--success), #86efac);
            }}

            .metric-label {{
                color: #94a3b8;
                font-size: .74rem;
                font-weight: 900;
                letter-spacing: .09em;
                text-transform: uppercase;
            }}

            .metric-value {{
                color: #f1f5f9;
                font-size: clamp(1.7rem, 2.3vw, 2.28rem);
                font-weight: 900;
                line-height: 1.1;
                margin-top: .52rem;
                white-space: normal;
                overflow-wrap: anywhere;
            }}

            .metric-caption {{
                color: #94a3b8;
                font-size: .85rem;
                margin-top: .55rem;
                line-height: 1.35;
            }}

            .section-title {{
                color: #f1f5f9;
                font-size: 1.28rem;
                font-weight: 850;
                margin: 1.35rem 0 .35rem;
            }}

            .section-note {{
                color: #94a3b8;
                font-size: .92rem;
                margin-bottom: .75rem;
            }}

            .section-block-title {{
                color: #f1f5f9;
                font-size: 1.32rem;
                font-weight: 900;
                line-height: 1.2;
                margin: 1.45rem 0 .35rem;
            }}

            .filter-caption {{
                color: #94a3b8;
                font-size: .88rem;
                font-weight: 700;
                margin-bottom: .35rem;
            }}

            .insight-card {{
                background: linear-gradient(180deg, rgba(30,41,59,.98), rgba(15,23,42,.96));
                border: 1px solid rgba(51,65,85,.6);
                border-left: 4px solid var(--primary);
                border-radius: 8px;
                padding: .95rem 1rem;
                box-shadow: 0 12px 30px rgba(0,0,0,.2);
                min-height: 112px;
            }}

            .insight-card.warning {{
                border-left-color: var(--danger);
            }}

            .insight-card.accent {{
                border-left-color: var(--accent);
            }}

            .insight-label {{
                color: #64748b;
                font-size: .72rem;
                font-weight: 850;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin-bottom: .35rem;
            }}

            .insight-value {{
                color: #f1f5f9;
                font-size: 1.18rem;
                font-weight: 850;
                line-height: 1.22;
            }}

            .insight-caption {{
                color: #94a3b8;
                font-size: .86rem;
                margin-top: .35rem;
            }}

            .kpi-heading-title {{
                color: #f1f5f9;
                font-size: 1.08rem;
                font-weight: 900;
                letter-spacing: 0;
            }}

            .kpi-heading-note {{
                color: #94a3b8;
                font-size: .88rem;
                font-weight: 700;
                text-align: right;
            }}

            .metric-card:hover {{
                transform: translateY(-3px);
                border-color: rgba(37,99,235,.32);
                box-shadow: 0 20px 44px rgba(15,23,42,.12);
            }}

            .metric-card::before {{
                content: "";
                position: absolute;
                inset: 0 0 auto 0;
                height: 4px;
                background: linear-gradient(90deg, var(--primary), var(--accent-soft));
            }}

            .metric-card.primary::before {{
                background: linear-gradient(90deg, var(--primary), #60a5fa);
            }}

            .metric-card.accent::before {{
                background: linear-gradient(90deg, var(--accent), var(--accent-soft));
            }}

            .metric-card.warning::before {{
                background: linear-gradient(90deg, var(--warning), #fcd34d);
            }}

            .metric-card.danger::before {{
                background: linear-gradient(90deg, var(--danger), #f43f5e);
            }}

            .metric-card.success::before {{
                background: linear-gradient(90deg, var(--success), #86efac);
            }}

            .metric-label {{
                color: var(--muted);
                font-size: .74rem;
                font-weight: 900;
                letter-spacing: .09em;
                text-transform: uppercase;
            }}

            .metric-value {{
                color: var(--ink);
                font-size: clamp(1.7rem, 2.3vw, 2.28rem);
                font-weight: 900;
                line-height: 1.1;
                margin-top: .52rem;
                white-space: normal;
                overflow-wrap: anywhere;
            }}

            .metric-caption {{
                color: var(--muted);
                font-size: .85rem;
                margin-top: .55rem;
                line-height: 1.35;
            }}

            .section-title {{
                color: var(--ink);
                font-size: 1.28rem;
                font-weight: 850;
                margin: 1.35rem 0 .35rem;
            }}

            .section-note {{
                color: var(--muted);
                font-size: .92rem;
                margin-bottom: .75rem;
            }}

            .section-block-title {{
                color: var(--ink);
                font-size: 1.32rem;
                font-weight: 900;
                line-height: 1.2;
                margin: 1.45rem 0 .35rem;
            }}

            .filter-caption {{
                color: var(--muted);
                font-size: .88rem;
                font-weight: 700;
                margin-bottom: .35rem;
            }}

            .insight-card {{
                background: linear-gradient(180deg, rgba(30,41,59,.98), rgba(15,23,42,.96));
                border: 1px solid rgba(51,65,85,.6);
                border-left: 4px solid var(--primary);
                border-radius: 8px;
                padding: .95rem 1rem;
                box-shadow: 0 12px 30px rgba(0,0,0,.2);
                min-height: 112px;
            }}

            .insight-card.warning {{
                border-left-color: var(--danger);
            }}

            .insight-card.accent {{
                border-left-color: var(--accent);
            }}

            .insight-label {{
                color: #64748b;
                font-size: .72rem;
                font-weight: 850;
                letter-spacing: .08em;
                text-transform: uppercase;
                margin-bottom: .35rem;
            }}

            .insight-value {{
                color: #f1f5f9;
                font-size: 1.18rem;
                font-weight: 850;
                line-height: 1.22;
            }}

            .insight-caption {{
                color: #94a3b8;
                font-size: .86rem;
                margin-top: .35rem;
            }}

            .download-row {{
                background: rgba(30,41,59,.9);
                border: 1px solid rgba(51,65,85,.6);
                border-radius: 8px;
                padding: .85rem 1rem;
                margin: .4rem 0 1rem;
                color: #cbd5e1;
            }}

            div[data-testid="stMetric"] {{
                background: linear-gradient(180deg, rgba(30,41,59,.98), rgba(15,23,42,.94));
                border: 1px solid rgba(51,65,85,.6);
                border-radius: 8px;
                padding: .9rem 1rem;
                box-shadow: 0 12px 30px rgba(0,0,0,.2);
            }}

            .stTabs [data-baseweb="tab-list"] {{
                gap: .5rem;
                margin-top: .35rem;
                margin-bottom: 1rem;
                padding-top: .25rem;
            }}

            .stTabs [data-baseweb="tab"] {{
                background: rgba(30,41,59,.9);
                border: 1px solid rgba(51,65,85,.6);
                border-radius: 8px;
                color: #e2e8f0;
                font-weight: 750;
                min-height: 44px;
                padding: .65rem 1.05rem;
                box-shadow: 0 8px 22px rgba(0,0,0,.15);
            }}

            .stTabs [aria-selected="true"] {{
                background: linear-gradient(135deg, #1e293b, #1e1b4b);
                border-color: rgba(59,130,246,.35);
                color: #60a5fa;
            }}

            .stButton > button,
            .stDownloadButton > button {{
                background: var(--primary);
                border: 0;
                border-radius: 8px;
                color: white;
                font-weight: 850;
                box-shadow: 0 12px 28px rgba(59,130,246,.18);
            }}

            div[data-testid="stDataFrame"] {{
                border: 1px solid rgba(51,65,85,.6);
                border-radius: 8px;
                overflow: hidden;
                box-shadow: 0 12px 30px rgba(0,0,0,.15);
            }}

            .download-row {{
                background: rgba(30,41,59,.9);
                border: 1px solid rgba(51,65,85,.6);
                border-radius: 8px;
                padding: .85rem 1rem;
                margin: .4rem 0 1rem;
                color: #cbd5e1;
            }}

            .footer {{
                color: #94a3b8;
                text-align: center;
                font-size: .85rem;
                padding: 1.2rem 0 .25rem;
            }}

            @media (max-width: 900px) {{
                .hero {{
                    padding: 1.75rem 1.35rem;
                }}

                .kpi-heading {{
                    align-items: flex-start;
                    flex-direction: column;
                    gap: .25rem;
                }}

                .kpi-heading-note {{
                    text-align: left;
                }}

                .metric-card {{
                    min-height: 132px;
                }}
            }}

        </style>
        """,
        unsafe_allow_html=True,
    )


def card(label: str, value: str, caption: str, tone: str = "primary") -> None:
    st.markdown(
        f"""
        <div class="metric-card {tone}">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def insight_card(label: str, value: str, caption: str, tone: str = "") -> None:
    st.markdown(
        f"""
        <div class="insight-card {tone}">
            <div class="insight-label">{escape(label)}</div>
            <div class="insight-value">{escape(value)}</div>
            <div class="insight-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(CHART_BG)
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=INK, pad=18)
    ax.set_xlabel(xlabel, color=MUTED, labelpad=9, fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel, color=MUTED, labelpad=9, fontsize=10, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.grid(axis="y", color=CHART_GRID, linewidth=1.0)
    ax.grid(axis="x", visible=False)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#334155")
    ax.margins(x=.03)


def show_chart(fig) -> None:
    fig.patch.set_facecolor(CHART_BG)
    fig.patch.set_alpha(1)
    fig.tight_layout(pad=1.35)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def plotly_style(fig, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "xanchor": "left", "font": {"size": 18, "color": INK}},
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font={"family": "Arial", "color": MUTED, "size": 12},
        margin={"l": 20, "r": 22, "t": 72, "b": 40},
        hoverlabel={
            "bgcolor": "#1e293b",
            "font_size": 12,
            "font_family": "Arial",
            "font_color": "white",
            "bordercolor": "#1e293b",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11, "color": MUTED},
        },
    )
    AXIS_LINE = "#334155"
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=AXIS_LINE,
        tickfont={"color": MUTED, "size": 11},
        title_font={"color": MUTED, "size": 12},
    )
    fig.update_yaxes(
        gridcolor=CHART_GRID,
        zeroline=False,
        linecolor=AXIS_LINE,
        tickfont={"color": MUTED, "size": 11},
        title_font={"color": MUTED, "size": 12},
    )
    return fig


def show_plotly(fig, title: str, height: int = 420) -> None:
    st.plotly_chart(plotly_style(fig, title, height), width="stretch", config=PLOTLY_CONFIG)


def plotly_empty(title: str, message: str) -> None:
    fig = go.Figure()
    fig.add_annotation(text=message, x=.5, y=.5, xref="paper", yref="paper", showarrow=False, font={"color": MUTED, "size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    show_plotly(fig, title)


def premium_figure(width: float = 7, height: float = 4.6):
    return plt.subplots(figsize=(width, height), facecolor=CHART_BG, dpi=FIG_DPI)


def label_bars(ax, horizontal: bool = False, percent_labels: bool = False) -> None:
    for container in ax.containers:
        labels = []
        for value in container.datavalues:
            labels.append(f"{value:.0%}" if percent_labels else f"{value:,.0f}")
        ax.bar_label(
            container,
            labels=labels,
            padding=5,
            fontsize=8.5,
            fontweight="bold",
            color=INK,
            label_type="edge",
        )
    if horizontal:
        ax.margins(x=.12)
    else:
        ax.margins(y=.14)


def soften_legend(ax) -> None:
    legend = ax.get_legend()
    if legend:
        legend.set_title("")
        legend.get_frame().set_facecolor("#f8fafc")
        legend.get_frame().set_edgecolor("#dbe3ee")
        legend.get_frame().set_linewidth(.8)
        for text in legend.get_texts():
            text.set_color(MUTED)
            text.set_fontsize(9)


def percent(value: float) -> str:
    if pd.isna(value):
        return "0.00%"
    return f"{value * 100:.2f}%"


def plot_empty(title: str, message: str) -> None:
    fig, ax = premium_figure()
    ax.text(.5, .5, message, ha="center", va="center", color=MUTED, fontsize=11)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=INK, pad=14)
    ax.set_axis_off()
    show_chart(fig)


def section_header(title: str, note: str = "") -> None:
    note_html = f'<div class="section-note">{escape(note)}</div>' if note else ""
    st.markdown(
        f"""
        <div class="section-block-title">{escape(title)}</div>
        {note_html}
        """,
        unsafe_allow_html=True,
    )


inject_css()
sns.set_theme(
    style="whitegrid",
    font="Arial",
    rc={
        "axes.facecolor": PANEL,
        "figure.facecolor": PANEL,
        "grid.color": GRID,
        "axes.edgecolor": "#cbd5e1",
    },
)
data = load_data(DATA_PATH)

gender_options = sorted(data["gender"].dropna().unique())
residence_options = sorted(data["Residence_type"].dropna().unique())
work_options = sorted(data["work_type"].dropna().unique())
marriage_options = sorted(data["ever_married"].dropna().unique())
smoking_options = sorted(data["smoking_status"].dropna().unique())

st.sidebar.markdown("## Dashboard Controls")
st.sidebar.caption("Use these filters to update the KPIs, charts, and data explorer.")

with st.sidebar:
    st.markdown("### Cohort")
    gender_filter = st.multiselect("Gender", options=gender_options, default=gender_options)
    residence_filter = st.multiselect("Residence type", options=residence_options, default=residence_options)
    work_filter = st.multiselect("Work type", options=work_options, default=work_options)
    marriage_filter = st.multiselect("Ever married", options=marriage_options, default=marriage_options)
    smoking_filter = st.multiselect("Smoking status", options=smoking_options, default=smoking_options)

    st.markdown("### Clinical")
    hypertension_filter = st.selectbox("Hypertension", options=["All", "No", "Yes"], index=0)
    heart_filter = st.selectbox("Heart disease", options=["All", "No", "Yes"], index=0)

    st.markdown("### Ranges")
    age_range = st.slider(
        "Age range",
        min_value=float(data["age"].min()),
        max_value=float(data["age"].max()),
        value=(float(data["age"].min()), float(data["age"].max())),
    )
    glucose_range = st.slider(
        "Average glucose range",
        min_value=float(data["avg_glucose_level"].min()),
        max_value=float(data["avg_glucose_level"].max()),
        value=(float(data["avg_glucose_level"].min()), float(data["avg_glucose_level"].max())),
    )
    bmi_range = st.slider(
        "BMI range",
        min_value=float(data["bmi"].min()),
        max_value=float(data["bmi"].max()),
        value=(float(data["bmi"].min()), float(data["bmi"].max())),
    )

st.markdown(
    """
    <section class="hero">
        <div class="hero-kicker">Clinical Analytics Dashboard</div>
        <h1>Stroke Risk Intelligence</h1>
        <p>
            A polished healthcare overview for exploring stroke indicators,
            patient demographics, glucose levels, BMI patterns, and risk signals.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

kpi_slot = st.empty()

filtered_data = data[
    data["gender"].isin(gender_filter)
    & data["smoking_status"].isin(smoking_filter)
    & data["Residence_type"].isin(residence_filter)
    & data["work_type"].isin(work_filter)
    & data["ever_married"].isin(marriage_filter)
    & data["age"].between(age_range[0], age_range[1])
    & data["avg_glucose_level"].between(glucose_range[0], glucose_range[1])
    & data["bmi"].between(bmi_range[0], bmi_range[1])
]

if hypertension_filter != "All":
    filtered_data = filtered_data[filtered_data["hypertension_label"] == hypertension_filter]

if heart_filter != "All":
    filtered_data = filtered_data[filtered_data["heart_disease_label"] == heart_filter]

total_patients = len(filtered_data)
stroke_cases = int(filtered_data["stroke"].sum()) if total_patients else 0
stroke_rate = filtered_data["stroke"].mean() if total_patients else 0
avg_age = filtered_data["age"].mean() if total_patients else 0
avg_glucose = filtered_data["avg_glucose_level"].mean() if total_patients else 0
avg_bmi = filtered_data["bmi"].mean() if total_patients else 0
avg_risk = filtered_data["risk_score"].mean() if total_patients else 0
elevated_risk_share = (filtered_data["risk_band"] == "Elevated").mean() if total_patients else 0
missing_bmi_share = filtered_data["bmi_missing"].mean() if total_patients else 0
active_filter_note = (
    f"Age {age_range[0]:.0f}-{age_range[1]:.0f} | "
    f"Glucose {glucose_range[0]:.0f}-{glucose_range[1]:.0f} | "
    f"BMI {bmi_range[0]:.1f}-{bmi_range[1]:.1f}"
)

with kpi_slot.container():
    st.markdown(
        f"""
        <div class="kpi-heading">
            <div class="kpi-heading-title">Executive KPI Strip</div>
            <div class="kpi-heading-note">{escape(active_filter_note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(5)
    with metric_cols[0]:
        card("Patients", f"{total_patients:,}", "Filtered records", "primary")
    with metric_cols[1]:
        card("Stroke Rate", percent(stroke_rate), f"{stroke_cases:,} stroke cases", "danger")
    with metric_cols[2]:
        card("Risk Score", f"{avg_risk:.1f}", "Composite 0-100 indicator", "warning")
    with metric_cols[3]:
        card("Elevated Risk", percent(elevated_risk_share), "Patients in top risk band", "accent")
    with metric_cols[4]:
        card("Average Age", f"{avg_age:.1f}", "Years", "success")

    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

if filtered_data.empty:
    st.warning("No records match the selected filters. Adjust the cohort controls to continue.")
    st.stop()

section_header(
    "Analytics Workspace",
    "Graphs and tables are separated into stable containers for a clean modern dashboard layout.",
)

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Risk Patterns", "Stratification", "Data"])

with tab1:
    st.markdown('<div class="section-title">Population Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">High-level composition and outcome distribution for the selected cohort.</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.2, 1])

    with left:
        with st.container(border=True):
            age_counts = filtered_data.groupby("age_group", observed=False).size().reset_index(name="patients")
            fig = px.bar(
                age_counts,
                x="age_group",
                y="patients",
                color="age_group",
                color_discrete_sequence=CLINICAL_PALETTE,
                text="patients",
                labels={"age_group": "Age group", "patients": "Patients"},
            )
            fig.update_traces(
                marker_line_color="white",
                marker_line_width=1.4,
                texttemplate="%{text:,}",
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Patients: %{y:,}<extra></extra>",
            )
            fig.update_layout(showlegend=False)
            show_plotly(fig, "Patients by Age Group")

    with right:
        with st.container(border=True):
            stroke_counts = filtered_data["stroke_label"].value_counts().reindex(["No Stroke", "Stroke"], fill_value=0)
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=stroke_counts.index,
                        values=stroke_counts.values,
                        hole=.62,
                        marker={"colors": [NO_STROKE, STROKE], "line": {"color": "white", "width": 4}},
                        textinfo="label+percent",
                        textfont={"size": 12, "color": INK},
                        hovertemplate="<b>%{label}</b><br>Patients: %{value:,}<br>Share: %{percent}<extra></extra>",
                    )
                ]
            )
            fig.add_annotation(text=f"<b>{total_patients:,}</b><br>patients", x=.5, y=.5, showarrow=False, font={"size": 17, "color": INK})
            show_plotly(fig, "Stroke Outcome Mix")

    st.markdown('<div class="section-title">Clinical Indicators</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Hypertension Share", percent(filtered_data["hypertension"].mean()))
    b.metric("Heart Disease Share", percent(filtered_data["heart_disease"].mean()))
    c.metric("Urban Residence Share", percent((filtered_data["Residence_type"] == "Urban").mean()))

    st.markdown('<div class="section-title">Fast Readout</div>', unsafe_allow_html=True)
    top_age_group = (
        filtered_data.groupby("age_group", observed=False)["stroke"]
        .mean()
        .sort_values(ascending=False)
        .dropna()
    )
    top_smoking_group = (
        filtered_data.groupby("smoking_status")["stroke"]
        .mean()
        .sort_values(ascending=False)
        .dropna()
    )
    i1, i2, i3 = st.columns(3)
    with i1:
        insight_card(
            "Highest age-group stroke rate",
            f"{top_age_group.index[0]} at {percent(top_age_group.iloc[0])}" if not top_age_group.empty else "Not available",
            "Within the currently filtered cohort.",
            "warning",
        )
    with i2:
        insight_card(
            "Most exposed smoking segment",
            f"{top_smoking_group.index[0]} at {percent(top_smoking_group.iloc[0])}" if not top_smoking_group.empty else "Not available",
            "Stroke rate by reported smoking status.",
            "accent",
        )
    with i3:
        insight_card(
            "BMI imputation footprint",
            percent(missing_bmi_share),
            "Records where missing BMI was filled with the median.",
        )

with tab2:
    st.markdown('<div class="section-title">Risk Pattern Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Compare demographics and clinical values against stroke outcomes.</div>',
        unsafe_allow_html=True,
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        with st.container(border=True):
            if len(filtered_data) >= 3 and filtered_data["stroke_label"].nunique() > 1:
                age_bins = list(range(0, 91, 10))
                age_labels = [f"{start}-{start + 9}" for start in age_bins[:-1]]
                age_distribution = filtered_data.assign(
                    age_band=pd.cut(filtered_data["age"], bins=age_bins, labels=age_labels, right=False, include_lowest=True)
                )
                age_distribution = (
                    age_distribution.groupby(["age_band", "stroke_label"], observed=False)
                    .size()
                    .reset_index(name="patients")
                )
                age_distribution["cohort_share"] = age_distribution["patients"] / total_patients
                fig = px.bar(
                    age_distribution,
                    x="age_band",
                    y="patients",
                    color="stroke_label",
                    barmode="group",
                    custom_data=["stroke_label", "patients", "cohort_share"],
                    color_discrete_map=OUTCOME_PALETTE,
                    labels={"age_band": "Age band", "patients": "Patients", "stroke_label": "Outcome"},
                )
                fig.update_traces(
                    marker_line_color="white",
                    marker_line_width=1.1,
                    hovertemplate=(
                        "<b>Age %{x}</b><br>"
                        "Outcome: %{customdata[0]}<br>"
                        "Patients: %{customdata[1]:,}<br>"
                        "Cohort share: %{customdata[2]:.1%}<extra></extra>"
                    ),
                )
                show_plotly(fig, "Age Distribution by Outcome")
            else:
                plotly_empty("Age Distribution by Outcome", "Widen filters to compare both outcomes.")

    with chart_col2:
        with st.container(border=True):
            fig = px.box(
                filtered_data,
                x="stroke_label",
                y="avg_glucose_level",
                color="stroke_label",
                points="suspectedoutliers",
                color_discrete_map=OUTCOME_PALETTE,
                labels={"stroke_label": "Outcome", "avg_glucose_level": "Average glucose level"},
            )
            fig.update_traces(marker={"size": 4, "opacity": .45}, line={"width": 2}, hovertemplate="<b>%{x}</b><br>Glucose: %{y:.1f}<extra></extra>")
            fig.update_layout(showlegend=False)
            show_plotly(fig, "Glucose Level by Outcome")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        with st.container(border=True):
            smoking_counts = (
                filtered_data.groupby(["smoking_status", "stroke_label"])
                .size()
                .reset_index(name="patients")
            )
            smoking_counts["cohort_share"] = smoking_counts["patients"] / total_patients
            fig = px.bar(
                smoking_counts,
                y="smoking_status",
                x="patients",
                color="stroke_label",
                orientation="h",
                barmode="group",
                text="patients",
                custom_data=["stroke_label", "patients", "cohort_share"],
                color_discrete_map=OUTCOME_PALETTE,
                labels={"smoking_status": "Smoking status", "patients": "Patients", "stroke_label": "Outcome"},
            )
            fig.update_traces(
                marker_line_color="white",
                marker_line_width=1.2,
                texttemplate="%{text:,}",
                textposition="outside",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Outcome: %{customdata[0]}<br>"
                    "Patients: %{customdata[1]:,}<br>"
                    "Cohort share: %{customdata[2]:.1%}<extra></extra>"
                ),
            )
            show_plotly(fig, "Smoking Status by Outcome")

    with chart_col4:
        with st.container(border=True):
            corr = filtered_data[
                ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "stroke"]
            ].corr(numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                color_continuous_scale="RdBu",
                zmin=-1,
                zmax=1,
                aspect="auto",
            )
            fig.update_traces(hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>")
            show_plotly(fig, "Correlation Matrix")

    st.markdown('<div class="section-title">Clinical Distribution</div>', unsafe_allow_html=True)
    chart_col5, chart_col6 = st.columns(2)

    with chart_col5:
        bmi_counts = filtered_data["bmi_category"].value_counts().reindex(
            ["Underweight", "Healthy", "Overweight", "Obese"], fill_value=0
        )
        bmi_plot = bmi_counts.reset_index()
        bmi_plot.columns = ["bmi_category", "patients"]
        bmi_plot["cohort_share"] = bmi_plot["patients"] / total_patients
        fig = px.bar(
            bmi_plot,
            x="bmi_category",
            y="patients",
            color="bmi_category",
            text="patients",
            custom_data=["patients", "cohort_share"],
            color_discrete_sequence=["#38bdf8", "#16a34a", "#f59e0b", "#dc2626"],
            labels={"bmi_category": "BMI category", "patients": "Patients"},
        )
        fig.update_traces(
            marker_line_color="white",
            marker_line_width=1.4,
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Patients: %{customdata[0]:,}<br>Cohort share: %{customdata[1]:.1%}<extra></extra>",
        )
        fig.update_layout(showlegend=False)
        show_plotly(fig, "BMI Category Mix", height=400)

    with chart_col6:
        glucose_counts = filtered_data["glucose_band"].value_counts().reindex(
            ["Normal", "Elevated", "High"], fill_value=0
        )
        glucose_plot = glucose_counts.reset_index()
        glucose_plot.columns = ["glucose_band", "patients"]
        glucose_plot["cohort_share"] = glucose_plot["patients"] / total_patients
        fig = px.bar(
            glucose_plot,
            x="glucose_band",
            y="patients",
            color="glucose_band",
            text="patients",
            custom_data=["patients", "cohort_share"],
            color_discrete_sequence=["#16a34a", "#f59e0b", "#dc2626"],
            labels={"glucose_band": "Glucose band", "patients": "Patients"},
        )
        fig.update_traces(
            marker_line_color="white",
            marker_line_width=1.4,
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Patients: %{customdata[0]:,}<br>Cohort share: %{customdata[1]:.1%}<extra></extra>",
        )
        fig.update_layout(showlegend=False)
        show_plotly(fig, "Glucose Band Mix", height=400)

with tab3:
    st.markdown('<div class="section-title">Risk Stratification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">A transparent composite score built from age, hypertension, heart disease, glucose, BMI, and smoking status.</div>',
        unsafe_allow_html=True,
    )

    risk_cols = st.columns(3)
    risk_counts = filtered_data["risk_band"].value_counts().reindex(["Low", "Watch", "Elevated"], fill_value=0)
    risk_rate = filtered_data.groupby("risk_band", observed=False)["stroke"].mean().reindex(["Low", "Watch", "Elevated"])
    with risk_cols[0]:
        insight_card("Low risk band", f"{risk_counts['Low']:,}", f"Stroke rate {percent(risk_rate['Low'])}", "accent")
    with risk_cols[1]:
        insight_card("Watch band", f"{risk_counts['Watch']:,}", f"Stroke rate {percent(risk_rate['Watch'])}")
    with risk_cols[2]:
        insight_card("Elevated band", f"{risk_counts['Elevated']:,}", f"Stroke rate {percent(risk_rate['Elevated'])}", "warning")

    st.markdown('<div class="kpi-spacer"></div>', unsafe_allow_html=True)

    risk_chart_1, risk_chart_2 = st.columns(2)
    with risk_chart_1:
        risk_count_plot = risk_counts.reset_index()
        risk_count_plot.columns = ["risk_band", "patients"]
        risk_count_plot["cohort_share"] = risk_count_plot["patients"] / total_patients
        fig = px.bar(
            risk_count_plot,
            x="risk_band",
            y="patients",
            color="risk_band",
            text="patients",
            custom_data=["patients", "cohort_share"],
            color_discrete_map=RISK_PALETTE,
            labels={"risk_band": "Risk band", "patients": "Patients"},
        )
        fig.update_traces(
            marker_line_color="white",
            marker_line_width=1.4,
            texttemplate="%{text:,}",
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Patients: %{customdata[0]:,}<br>Cohort share: %{customdata[1]:.1%}<extra></extra>",
        )
        fig.update_layout(showlegend=False)
        show_plotly(fig, "Patients by Risk Band")

    with risk_chart_2:
        risk_plot = risk_rate.reset_index(name="stroke_rate")
        risk_plot["patients"] = risk_plot["risk_band"].map(risk_counts).fillna(0).astype(int)
        risk_cases = filtered_data.groupby("risk_band", observed=False)["stroke"].sum().reindex(["Low", "Watch", "Elevated"], fill_value=0)
        risk_plot["stroke_cases"] = risk_plot["risk_band"].map(risk_cases).fillna(0).astype(int)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=risk_plot["risk_band"],
                y=risk_plot["stroke_rate"],
                customdata=risk_plot[["stroke_cases", "patients"]].to_numpy(),
                mode="lines+markers+text",
                line={"color": STROKE, "width": 4, "shape": "spline"},
                marker={"size": 11, "color": STROKE, "line": {"color": "white", "width": 2}},
                fill="tozeroy",
                fillcolor="rgba(220,38,38,.12)",
                text=[percent(v) for v in risk_plot["stroke_rate"].fillna(0)],
                textposition="top center",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Stroke rate: %{y:.2%}<br>"
                    "Stroke cases: %{customdata[0]:,}<br>"
                    "Patients: %{customdata[1]:,}<extra></extra>"
                ),
            )
        )
        fig.update_yaxes(tickformat=".0%")
        show_plotly(fig, "Observed Stroke Rate by Risk Band")

    heat_counts = pd.crosstab(filtered_data["age_group"], filtered_data["risk_band"]).reindex(
        index=["0-18", "19-35", "36-50", "51-65", "66+"],
        columns=["Low", "Watch", "Elevated"],
        fill_value=0,
    )
    heat = pd.crosstab(filtered_data["age_group"], filtered_data["risk_band"], normalize="index").reindex(
        index=["0-18", "19-35", "36-50", "51-65", "66+"],
        columns=["Low", "Watch", "Elevated"],
        fill_value=0,
    )
    fig = go.Figure(
        data=[
            go.Heatmap(
                z=heat.values,
                x=heat.columns.astype(str),
                y=heat.index.astype(str),
                customdata=heat_counts.values,
                colorscale=[[0, "#eff6ff"], [1, "#1e40af"]],
                colorbar={"tickformat": ".0%", "title": "Share"},
                text=[[f"{value:.0%}" for value in row] for row in heat.values],
                texttemplate="%{text}",
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Risk band: %{x}<br>"
                    "Share of age group: %{z:.1%}<br>"
                    "Patients: %{customdata:,}<extra></extra>"
                ),
            )
        ]
    )
    show_plotly(fig, "Risk Band Mix by Age Group", height=430)

with tab4:
    st.markdown('<div class="section-title">Patient-Level Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Search, sort, and export-ready table view of the filtered records.</div>',
        unsafe_allow_html=True,
    )

    display_columns = [
        "gender",
        "age",
        "hypertension_label",
        "heart_disease_label",
        "ever_married",
        "work_type",
        "Residence_type",
        "avg_glucose_level",
        "bmi",
        "bmi_category",
        "glucose_band",
        "risk_score",
        "risk_band",
        "smoking_status",
        "stroke_label",
    ]
    export_data = filtered_data[display_columns].rename(
        columns={
            "gender": "Gender",
            "age": "Age",
            "hypertension_label": "Hypertension",
            "heart_disease_label": "Heart Disease",
            "ever_married": "Ever Married",
            "work_type": "Work Type",
            "Residence_type": "Residence",
            "avg_glucose_level": "Avg Glucose",
            "bmi": "BMI",
            "bmi_category": "BMI Category",
            "glucose_band": "Glucose Band",
            "risk_score": "Risk Score",
            "risk_band": "Risk Band",
            "smoking_status": "Smoking Status",
            "stroke_label": "Stroke Outcome",
        }
    )
    st.markdown(
        '<div class="download-row">Download the currently filtered table for handoff or deeper analysis.</div>',
        unsafe_allow_html=True,
    )
    st.download_button(
        "Download filtered CSV",
        data=export_data.to_csv(index=False).encode("utf-8"),
        file_name="filtered_stroke_risk_cohort.csv",
        mime="text/csv",
        width="stretch",
    )
    st.dataframe(
        export_data,
        width="stretch",
        hide_index=True,
    )

st.markdown(
    '<div class="footer">Dashboard created with Streamlit for healthcare analytics and cohort exploration.</div>',
    unsafe_allow_html=True,
)

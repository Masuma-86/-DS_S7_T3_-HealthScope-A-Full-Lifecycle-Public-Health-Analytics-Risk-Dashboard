"""
Exploratory Data Analysis (EDA) module for HealthScope multi-disease analytics platform.
Generates clean visual charts for Diabetes, Heart Disease, and Hypertension datasets.
Theme Palette: Primary Navy (#1E3A8A), Secondary Teal (#0D9488), Coral Accent (#E11D48).
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

NAVY = "#1E3A8A"
TEAL = "#0D9488"
CORAL = "#E11D48"
BG_COLOR = "#FFFFFF"


def set_chart_style():
    """Sets clean theme styling for all Matplotlib/Seaborn figures."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        'figure.facecolor': BG_COLOR,
        'axes.facecolor': BG_COLOR,
        'axes.edgecolor': '#E2E8F0',
        'axes.labelcolor': '#0F172A',
        'xtick.color': '#475569',
        'ytick.color': '#475569',
        'grid.color': '#F1F5F9',
        'grid.linestyle': '--',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.titlecolor': NAVY
    })


def plot_outcome_distribution(df, target_col, disease_name):
    """Plot disease outcome distribution (0 vs 1)."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    counts = df[target_col].value_counts()
    
    labels = [f'Negative (0)', f'Positive (1)']
    colors = [TEAL, NAVY]
    
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors, width=0.45, edgecolor='#CBD5E1')
    total = len(df)
    for bar in bars:
        height = bar.get_height()
        pct = (height / total) * 100
        ax.annotate(f'{height}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', color='#1E293B')

    ax.set_ylabel('Patient Count')
    ax.set_title(f'{disease_name} Target Distribution')
    ax.set_ylim(0, max(counts) * 1.18)
    plt.tight_layout()
    return fig


def plot_feature_distribution(df, feature_col, title_label=None, color=NAVY):
    """Histogram & KDE plot for numerical continuous features."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    clean_series = df[feature_col].dropna()
    sns.histplot(clean_series, kde=True, color=color, bins=20, ax=ax, edgecolor='white')
    ax.set_xlabel(title_label or feature_col)
    ax.set_ylabel('Patient Count')
    ax.set_title(f'Distribution of {title_label or feature_col}')
    plt.tight_layout()
    return fig


def plot_feature_vs_outcome(df, feature_col, target_col, feature_name=None, disease_name=""):
    """Boxplot comparing continuous feature distribution by disease outcome."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    clean_df = df[[feature_col, target_col]].dropna().copy()
    clean_df['Outcome_Label'] = clean_df[target_col].map({0: 'Negative', 1: 'Positive'})

    sns.boxplot(
        data=clean_df, x='Outcome_Label', y=feature_col,
        palette={'Negative': TEAL, 'Positive': CORAL},
        width=0.4, ax=ax, boxprops=dict(alpha=0.85)
    )
    ax.set_xlabel('Diagnosis Status')
    ax.set_ylabel(feature_name or feature_col)
    ax.set_title(f'{feature_name or feature_col} by {disease_name} Outcome')
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df, feature_cols, target_col, disease_name=""):
    """Pearson correlation matrix heatmap."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    
    cols = feature_cols + [target_col]
    corr = df[cols].dropna().corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr, mask=mask, cmap='Blues', annot=True, fmt=".2f",
        linewidths=0.5, ax=ax, cbar_kws={"shrink": .8}, square=True,
        annot_kws={"size": 9}
    )
    ax.set_title(f'{disease_name} Correlation Heatmap')
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importance_scores, title="Feature Importance"):
    """Bar chart displaying model feature importance scores."""
    set_chart_style()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_scores})
    fi_df = fi_df.sort_values(by='Importance', ascending=True)

    ax.barh(fi_df['Feature'], fi_df['Importance'], color=TEAL, edgecolor='#CBD5E1', height=0.6)
    ax.set_xlabel('Importance Score')
    ax.set_title(title)
    plt.tight_layout()
    return fig

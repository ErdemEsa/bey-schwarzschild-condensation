"""
Schwarzschild Condensation Project
Unified plotting configuration for publication-quality figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# ============================================================
# EPJC-compatible figure settings
# ============================================================

COLUMN_WIDTH = 3.4     # inches (single column)
TEXT_WIDTH = 7.0        # inches (double column)
GOLDEN_RATIO = (1 + np.sqrt(5)) / 2

# Color palette (colorblind-friendly)
COLORS = {
    'primary':    '#2E86AB',   # Blue
    'secondary':  '#A23B72',   # Purple
    'tertiary':   '#F18F01',   # Orange
    'quaternary': '#C73E1D',   # Red
    'quinary':    '#3B1F2B',   # Dark
    'success':    '#2E8B57',   # Green
    'gray':       '#7F7F7F',   # Gray
    'light_gray': '#D0D0D0',   # Light gray
}

def setup_plots():
    """Configure matplotlib for EPJC publication quality."""
    plt.rcParams.update({
        # Font
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'],
        'text.usetex': True,
        'text.latex.preamble': r'\usepackage{amsmath}\usepackage{amssymb}',
        
        # Font sizes
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        
        # Figure
        'figure.figsize': (COLUMN_WIDTH, COLUMN_WIDTH / GOLDEN_RATIO),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.03,
        
        # Axes
        'axes.linewidth': 0.6,
        'axes.grid': False,
        
        # Ticks
        'xtick.major.size': 3,
        'xtick.minor.size': 1.5,
        'xtick.major.width': 0.5,
        'xtick.minor.width': 0.4,
        'xtick.direction': 'in',
        'ytick.major.size': 3,
        'ytick.minor.size': 1.5,
        'ytick.major.width': 0.5,
        'ytick.minor.width': 0.4,
        'ytick.direction': 'in',
        
        # Lines
        'lines.linewidth': 1.0,
        'lines.markersize': 3,
        
        # Legend
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': 'none',
        'legend.fancybox': False,
    })

def save_figure(fig, name, formats=['pdf', 'png']):
    """Save figure in multiple formats."""
    fig_dir = '/workspace/figures'
    os.makedirs(fig_dir, exist_ok=True)
    
    for fmt in formats:
        filepath = os.path.join(fig_dir, f'{name}.{fmt}')
        fig.savefig(filepath)
        print(f"  Saved: {filepath}")

def single_column_fig(height_ratio=None):
    """Create a single-column figure."""
    if height_ratio is None:
        height_ratio = 1.0 / GOLDEN_RATIO
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * height_ratio))
    return fig, ax

def double_column_fig(height_ratio=None):
    """Create a double-column figure."""
    if height_ratio is None:
        height_ratio = 0.4
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH, TEXT_WIDTH * height_ratio))
    return fig, ax

def double_panel_fig(height_ratio=None):
    """Create a double-panel (side by side) figure."""
    if height_ratio is None:
        height_ratio = 0.4
    fig, (ax1, ax2) = plt.subplots(1, 2, 
        figsize=(TEXT_WIDTH, TEXT_WIDTH * height_ratio))
    return fig, (ax1, ax2)

# Initialize on import
setup_plots()
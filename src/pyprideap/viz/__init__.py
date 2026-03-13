"""Visualization sub-package — plots, themes, and QC reports."""

from pyprideap.viz.theme import (
    PRIDE_COLORS,
    pride_color_discrete,
    pride_color_gradient,
    pride_fill_discrete,
    set_plot_theme,
)

__all__ = [
    "PRIDE_COLORS",
    "pride_color_discrete",
    "pride_color_gradient",
    "pride_fill_discrete",
    "set_plot_theme",
]

"""
Common font style of all figures
================================

All figures are saved at 300 dpi, so one font size in points gives the same
letter height in every panel of a composite figure. The plotting functions set
some font sizes themselves (and seaborn changes them in the boxen plots of f05);
:func:`style_fonts` is therefore applied to every figure just before it is saved
and sets each kind of text to the size below.

    FONT_FAMILY, FONT_SIZES   font and sizes [pt] for each kind of text
    FONT_RC                   matching matplotlib rcParams (set in the notebooks)
    style_fonts               apply family and sizes to all texts of a figure
"""

FONT_FAMILY = "DejaVu Sans"
FONT_SIZES = {
    "tick": 13,          # tick labels (axes and colour bars)
    "label": 14,         # axis and colour-bar labels, figure-level x/y labels
    "title": 15,         # panel titles (bold)
    "legend": 12,        # legend entries
    "annotation": 12,    # text in the panels (n=..., r=...)
    "count_tick": 11,    # tick labels of the grey "number of events" axes
    "count_label": 12,   # label of the grey "number of events" axes
}
FONT_RC = {
    "font.family": FONT_FAMILY,
    "mathtext.fontset": "dejavusans",
    "font.size": FONT_SIZES["tick"],
    "axes.labelsize": FONT_SIZES["label"],
    "axes.titlesize": FONT_SIZES["title"],
    "axes.titleweight": "bold",
    "xtick.labelsize": FONT_SIZES["tick"],
    "ytick.labelsize": FONT_SIZES["tick"],
    "legend.fontsize": FONT_SIZES["legend"],
    "figure.titlesize": FONT_SIZES["title"],
    "figure.labelsize": FONT_SIZES["label"],
}


def _is_count_axis(ax):
    """Secondary y axis with the number of events per time step ('n' or 'Valid events / bin')."""
    label = ax.get_ylabel().strip()
    return label == "n" or label.startswith("Valid events")


def style_fonts(fig, tight=False):
    """Set the font family and the font size of every text in a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to restyle (all axes, colour bars, legends and figure texts).
    tight : bool
        Re-run ``fig.tight_layout()`` afterwards (for figures that were laid out
        with tight_layout; constrained-layout figures adjust when they are saved).
    """
    for ax in fig.axes:
        count_axis = _is_count_axis(ax)
        tick = FONT_SIZES["count_tick" if count_axis else "tick"]
        label = FONT_SIZES["count_label" if count_axis else "label"]
        ax.tick_params(axis="both", which="both", labelsize=tick, labelfontfamily=FONT_FAMILY)
        for text in (ax.xaxis.label, ax.yaxis.label):
            text.set_fontsize(label)
            text.set_fontfamily(FONT_FAMILY)
        for text in (ax.xaxis.get_offset_text(), ax.yaxis.get_offset_text()):
            text.set_fontsize(tick)
        for text in (ax.title, ax._left_title, ax._right_title):
            if text.get_text():
                text.set_fontsize(FONT_SIZES["title"])
                text.set_fontweight("bold")
                text.set_fontfamily(FONT_FAMILY)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts() + [legend.get_title()]:
                text.set_fontsize(FONT_SIZES["legend"])
                text.set_fontfamily(FONT_FAMILY)
        for text in ax.texts:                       # also ax.annotate(...)
            text.set_fontsize(FONT_SIZES["annotation"])
            text.set_fontfamily(FONT_FAMILY)
    for legend in fig.legends:
        for text in legend.get_texts() + [legend.get_title()]:
            text.set_fontsize(FONT_SIZES["legend"])
            text.set_fontfamily(FONT_FAMILY)
    for text in fig.texts:                          # suptitle, supxlabel, supylabel
        if text.get_text():
            text.set_fontsize(FONT_SIZES["label"])
            text.set_fontfamily(FONT_FAMILY)
    if tight:
        fig.tight_layout()
    return fig

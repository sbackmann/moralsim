import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, FormatStrFormatter
import seaborn as sns


def wide_to_long(df_wide, value_name):
    """
    df_wide: DataFrame indexed by scenario, columns=models
    value_name: e.g. "score"
    """
    return (
        df_wide
          .reset_index()
          .melt(id_vars="index",
                var_name="model",
                value_name=value_name)
          .rename(columns={"index":"scenario"})
    )


def grouped_bar_from_wide(
    mean_wide,
    secondary_wide=None,
    stack_scenarios: list[str] = None,
    models=None,
    scenario_order=None,
    *,
    ax=None,
    scenario_labels: dict[str,str] | None = None,
    palette="tab10",
    group_width=0.8,
    figsize=(5,3),
    fontsize=7,
    dpi=300,
    ylabel=None,
    xlabel=None
):
    """
    Draw grouped bars for `mean_wide` (index=models, cols=scenarios).
    You can also subset to a list of `models` or to a `scenario_order`.
    """
    # determine models & scenarios
    if models is None:
        models = list(mean_wide.index)
    if scenario_order is None:
        scenario_order = list(mean_wide.columns)
    if scenario_labels is None:
        scenario_labels = {sc: sc for sc in scenario_order}

    models_print = models
    models = [model.replace("\n", "-") for model in models]

    # subset & align
    mdf = mean_wide.loc[models, scenario_order]

    n_mod  = len(models)
    n_scen = len(scenario_order)
    # figure & axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure
    if ax is None:
        fig.set_size_inches(*figsize)
        fig.set_dpi(dpi)

    num_colors = n_scen if stack_scenarios is None else n_scen + len(stack_scenarios)
    colors = sns.color_palette(palette, n_scen)
    ax.set_prop_cycle("color", colors)
    ax.set_xticks(np.arange(n_mod))
    ax.set_xticklabels(models_print, rotation=0, ha="center", fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(axis="y", linestyle=":", linewidth=0.3)

    bar_width = group_width / n_scen
    


    for i, scen in enumerate(scenario_order):
        xs = np.arange(n_mod) + (i - (n_scen-1)/2) * bar_width
        ys = mdf[scen].values
        label = scenario_labels.get(scen, scen)

        # decide if this scenario gets the “stacked two‐segment” treatment
        do_stack = (stack_scenarios is not None and scen in stack_scenarios
                    and secondary_wide is not None)

        if do_stack:
            # lower segment
            zs = secondary_wide.loc[models, scen].values
            ax.bar(
                xs, zs,
                width=bar_width,
                label=label,
                edgecolor="black", linewidth=0.3,
                color=colors[1],
                alpha=1
            )
            # upper segment
            ax.bar(
                xs, ys-zs,
                bottom=zs,
                width=bar_width,
                label=None,
                edgecolor="black", linewidth=0.3,
                color=colors[1],
                alpha=0.7
            )
        else:
            ax.bar(
                xs, ys, width=bar_width, label=label,
                edgecolor="black", linewidth=0.3
            )
    leg = ax.legend(fontsize=fontsize, title_fontsize=8, frameon=False, loc="upper center", ncol=len(scenario_order), handletextpad=0.5, columnspacing=1.2)
    return fig, ax
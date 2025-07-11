from collections import defaultdict
import datetime
from pathlib import Path
import sys
import itertools as it
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patheffects
from matplotlib.ticker import LogLocator, ScalarFormatter
import numpy as np
import pandas as pd
import seaborn as sns

# styles
BASIC = "basic"
BAD_INIT = "bad_init"
FAN = "fan"
WEIGHT = "weight"
MULTI_PARAM = "multi_param"
EPISODIC = "episodic"
STYLES = [BASIC, BAD_INIT, FAN, WEIGHT, MULTI_PARAM, EPISODIC]

# column names
TIME = "time (sec)"
ERR = "tracking error (cm)"
COST_CUM = "cumulative cost"
REGRET = "regret vs. expert"
EXPERIMENT = "scenario"
RATIO_DEFAULT = r"value / init"

# optimizer names
EXPERT = "expert"
DETUNE = "detune"
GAPS = "M-GAPS"
EPISODIC = "DiffTune"
EPISODIC_STAR = EPISODIC + r"$\star$"
SINGLEPOINT = "OPRF"
OPT_ORDER = [EXPERT, DETUNE, GAPS, EPISODIC_STAR, EPISODIC, SINGLEPOINT]
OPT_ORDER_COST = [DETUNE, SINGLEPOINT, EPISODIC, EPISODIC_STAR, GAPS, EXPERT]

# other constants
GAINTYPES = ["ki", "kp", "kv", "kr", "kw"]
GAINTYPES_DISPLAY = ["$k_i$", "$k_p$", "$k_v$", "$k_r$", r"$k_\omega$"]
GAIN2DISPLAY = dict(zip(GAINTYPES, GAINTYPES_DISPLAY))
AXES = ["xy", "z"]
GAPS_COLOR = "#0081EA"
EPISODIC_COLOR = [1.0, 0.7, 0.2]
EXPERT_COLOR = "#000000"



def agg(series):
    if series.dtype == "object":
        return series.iloc[0]
    return series.mean()


def normalize(x):
    return x / np.linalg.norm(x)


def plot_colorchanging(ax, x, y, maxtime, *args, **kwargs):
    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segments,
        path_effects=[patheffects.Stroke(capstyle="round")],
        *args, **kwargs
    )
    # Set the values used for colormapping
    lc.set_array(np.linspace(0, maxtime, len(x)))
    ax.add_collection(lc)
    return lc


def shade_fan(df, ax):
    fan_state = df["fan"] == 1
    fan_state = fan_state.bfill().ffill()
    fan_state.iloc[-1] = False
    fan_toggles = np.flatnonzero(np.diff(fan_state) != 0)
    assert len(fan_toggles) & 0x1 == 0
    # convert from index to time
    fan_toggles = df[TIME].iloc[fan_toggles].to_numpy()
    label = "fan on"
    handle = None
    for pair in fan_toggles.reshape((-1, 2)):
        handle = ax.axvspan(*pair, alpha=0.12, color="black", linewidth=0, label=label)
        label = None
    return handle


def planar_traj_coords(planar_trajectory, updir):
    # Note: trajectory must be EXACTLY planar. Not doing any least-squares.
    n, d = planar_trajectory.shape
    if len(updir) != d:
        raise ValueError("dimensions don't match")
    # make sure norm of cross product is meaningful
    norm_threshold = 1e-1 * np.mean(planar_trajectory.flat)
    normal = np.zeros(d)
    while np.linalg.norm(normal) < norm_threshold:
        a, b, c = planar_trajectory[np.random.choice(n, size=3)]
        normal = np.cross(a - c, b - c)
    if np.dot(normal, updir) < 0:
        normal = -normal
    normal = normalize(normal)
    print("normal is", normal)
    x = normalize(np.cross(updir, normal))
    y = np.cross(normal, x)
    M = np.stack([x, y])
    assert M.shape == (2, d)
    assert np.allclose(M @ M.T, np.eye(2))
    return M


def plot_fig8(dfs, style):

    sns.set_style("whitegrid")

    dfs = [df for df in dfs if df["trial"][0] == 1]
    dfs = sorted(dfs, key=lambda df: OPT_ORDER.index(df["optimizer"][0]))

    width = len(dfs) * (1.5 if style == FAN else 2.3)
    fig_fig8, axs_fig8 = plt.subplots(
        1, len(dfs),
        figsize=(width, 2.1),
        constrained_layout=True,
        sharey=True,
    )

    target_cols = ["target_" + c for c in "xyz"]
    pos_cols = ["pos_" + c for c in "xyz"]
    keep_cols = [TIME] + target_cols + pos_cols

    transform = None

    for ax, df in zip(axs_fig8, dfs):

        name = df["optimizer"].iloc[0]

        # The 10ms interp is a bit slow, so only grab the columns we need.
        maxtime = df[TIME].max()
        maxtime_i = int(maxtime + 0.5)
        if np.isclose(maxtime, maxtime_i, atol=0.1):
            maxtime = maxtime_i
        df = df[keep_cols].copy()
        df[TIME] = pd.to_timedelta(df[TIME], unit="seconds")
        df = df.set_index(TIME)
        # NOTE: 20ms is the longest possible interval, then jaggies appear.
        df = df.resample("20ms").apply(agg)
        target_mask = df.index < datetime.timedelta(seconds=4.0)

        target = np.stack([df[c][target_mask] for c in target_cols], axis=1)
        pos = np.stack([df[c] for c in pos_cols], axis=1)

        if False:
            assert style != FAN  # TODO: handle plane normal
            # plot in the basis of the trajectory plane instead of x/z.
            # currently disabled because it doesn't change the appearance much,
            # and it's easier to explain x/z.
            if transform is None:
                transform = planar_traj_coords(target, updir=(0, 0, 1))
                print("basis:", transform)
                transform = np.eye(3)[[0, 2], :]
            target = target @ transform.T
            pos = pos @ transform.T
            assert pos.shape[-1] == 2
        else:
            if style == FAN:
                pos = pos[:, [1, -1]]
                target = target[:, [1, -1]]
            else:
                ctr = np.mean(target, axis=0)
                shift = -ctr[[0, 2]]
                pos = pos[:, [0, -1]] + shift
                target = target[:, [0, -1]] + shift

        ax.plot(*target.T, label="target", linestyle=":", color="gray", linewidth=1.5)

        cmap = "coolwarm"
        line = plot_colorchanging(
            ax, *pos.T, maxtime=maxtime,
            label=name, cmap=cmap, linewidth=2,
        )
        ax.set(title=name, xlabel="horizontal (m)", ylabel="vertical (m)")
        ax.axis("equal")
        ax.set(xlim=[-0.8, 0.8])
        ax.set_xticks([-0.5, 0, 0.5])
        ax.set_xticks([-0.75, -0.25, 0.25, 0.75], minor=True)
        ax.grid(True, which="both")
        ax.set(ylim=[-0.55, 0.55], yticks=[-0.5, -0.25, 0, 0.25, 0.5])

        if ax != axs_fig8[0]:
            ax.set_ylabel(None)

        sns.despine(ax=ax, left=True, bottom=True)

    cbar = fig_fig8.colorbar(line, ticks=[0, maxtime / 2, maxtime])
    cbar.ax.set_ylabel(TIME)
    fig_fig8.savefig(f"{style}_fig8.pdf")


def fan_plot_laps(dfs: Sequence[pd.DataFrame]):
    dfs_laps = [
        df.resample("4s").apply(agg).reset_index()
        for df in dfs
    ]
    df = pd.concat(dfs_laps).reset_index()

    grid = sns.relplot(
        df,
        kind="line",
        markers=True,
        x=TIME,
        y=ERR,
        hue="optimizer",
        hue_order=[EXPERT, GAPS],
        palette=[EXPERT_COLOR, GAPS_COLOR],
        height=2.0,
        aspect=2.0,
    )

    tmax = df[TIME].max()
    ymax = grid.axes.flat[0].get_ylim()[1]
    lap_ticks = np.array([1, 12, 24, 36])
    lap_tick_times = 4 * (lap_ticks - 1) + 2
    grid.set(
        xticks=lap_tick_times,
        xticklabels=[f"${x}$" for x in lap_ticks],
        xlim=[2, tmax + 0.5],
        ylim=[0, ymax],
        xlabel="lap",
        ylabel="mean error (cm)",
    )

    sns.move_legend(grid, loc="lower left", bbox_to_anchor=(0.83, 0.25))
    handle = shade_fan(dfs[0], grid.axes.flat[0])
    grid.add_legend({"fan on": handle}, loc="upper left", bbox_to_anchor=(0.83, 0.9))

    grid.savefig("fan_laps.pdf")


def plot_costs(dfs: Sequence[pd.DataFrame], style):

    sns.set_style("whitegrid")

    optimizer_styles = {
        GAPS: dict(color="black"),
        EXPERT: dict(color=(0, 0.8, 0.4)),
        DETUNE: dict(color=(1.0, 0.2, 0.4)),
        SINGLEPOINT: dict(color=(0, 0.8, 1.0)),
        EPISODIC: dict(color=(0.6, 0.1, 0.8), linestyle=":"),
        EPISODIC_STAR: dict(color=(0.6, 0.1, 0.8), linestyle=None),
    }

    # take downsampled means to smooth the plots a little.
    dfs_sampled = []
    for df in dfs:
        df["timedelta"] = pd.to_timedelta(df[TIME], unit="seconds")
        dfi = df.set_index("timedelta")
        keep_cols = [TIME, "cost", ERR, "optimizer"]
        for optional_key in ["trial", "fan"]:
            if optional_key in dfi.columns:
                keep_cols.append(optional_key)
        dfi = dfi[keep_cols]
        dfr = dfi.resample("100ms").apply(agg)
        dfr[TIME] = dfr.index.total_seconds()
        # this used to be before resampling, but that was wrong!
        dfr[COST_CUM] = (dfr["cost"] * dfr[TIME].diff()).cumsum()
        dfs_sampled.append(dfr)
    dfs = dfs_sampled

    # TODO: figure out a more SQL-y way to do this. Ideally we wouldn't even
    # need the dataframe split.
    dfs_base = [df for df in dfs if df["optimizer"].iloc[0] == EXPERT]
    regret_baseline = (1 / len(dfs_base)) * sum(df[COST_CUM] for df in dfs_base)
    # NOTE: below is somewhat logical but let's stick with "expected" cost
    #regret_baseline = min(*dfs_base, key=lambda df: df[COST_CUM][-1])[COST_CUM]
    for df in dfs:
        df[REGRET] = df[COST_CUM] - regret_baseline

    df = pd.concat(dfs).reset_index()

    if style == FAN:
        fan_plot_laps(dfs)
    elif style == WEIGHT:
        df = pd.concat(dfs).reset_index()

        grid = sns.relplot(
            df,
            kind="line",
            markers=True,
            x=TIME,
            y=ERR,
            hue="optimizer",
            hue_order=[EXPERT, GAPS],
            palette=[EXPERT_COLOR, GAPS_COLOR],
            height=2.0,
            aspect=2.0,
        )

        tmax = df[TIME].max()
        ymax = grid.axes.flat[0].get_ylim()[1]
        grid.set(
            xlim=[0, tmax],
            ylim=[0, ymax],
        )
        grid.savefig("weight_cost.pdf")
        
    elif style != BAD_INIT:
        fig, axs = plt.subplots(1, 2, figsize=(5, 1.7), constrained_layout=True)
        ax_err, ax_regret = axs
        tmax = df[TIME].max()
        sns.lineplot(
            df,
            ax=ax_regret,
            x=TIME,
            y=REGRET,
            hue="optimizer",
            hue_order=OPT_ORDER_COST,
            errorbar="sd",
            legend=False,
        )
        ax_regret.set(xlim=[0, tmax])
        for i, opt in enumerate(OPT_ORDER_COST):
            z = 1000 - i  # on top of grid, etc
            opt_dfs = [df for df in dfs if df["optimizer"][0] == opt]
            label = opt
            for df in opt_dfs:
                #ax_regret.plot(df[TIME], df[REGRET], label=label, zorder=z, **optimizer_styles[opt])
                dflaps = df.resample("4s").apply(agg).reset_index()
                sns.lineplot(
                    dflaps,
                    ax=ax_err,
                    x=TIME,
                    y=ERR,
                    hue="optimizer",
                    hue_order=OPT_ORDER_COST,
                    legend=False,
                )
                # ax_err.plot(
                #     xticks,
                #     dflaps[ERR],
                #     label=label,
                #     zorder=z,
                #     marker=".",
                #     linewidth=1,
                #     markersize=10,
                #     **optimizer_styles[opt]
                # )
                # label = None
        # ax_err.set(xticks=xticks, xlabel="lap", ylabel=ERR)
        lap_ticks = np.array([1, 12, 24, 36])
        lap_tick_times = 4 * (lap_ticks - 1) + 2
        ax_err.set(
            xticks=lap_tick_times,
            xticklabels=lap_ticks,
            xlim=[2, tmax-1.5],
            xlabel="lap",
            ylabel="mean error (cm)",
        )
        if style == BAD_INIT:
            ax_regret.set_ylim([-0.03, 0.6])
            ax_regret.set(xticks=np.linspace(0, 32, 5), xlim=(0, 32))

        if style == FAN:
            for ax in axs:
                shade_fan(dfs[0], ax)
            # TODO: restore fan to legend!!

        fig.savefig(f"{style}_cost.pdf")
    else:
        kws = {}
        if len(df["optimizer"].unique()) > 2:
            kws["hue_order"] = OPT_ORDER_COST
        if "trial" in df.columns and len(df["trial"].unique()) > 1:
            kws["errorbar"] = "sd"

        grid = sns.relplot(
            df,
            kind="line",
            x=TIME,
            y=REGRET,
            hue="optimizer",
            height=3.0,
            aspect=1.4,
            **kws,
        )
        if style == BAD_INIT:
            grid.set(ylim=[-0.03, 0.5], xticks=np.linspace(0, 24, 7), xlim=(0, 24))
        grid.savefig(f"{style}_cost.pdf")


def param_format(p):
    """Converts code-style names for controller parameters to LaTeX."""
    if p[0] != "k" or p[2] != "_":
        return p
    kind = p[1]
    axis = p[3:]
    return "$k_{%s}^{%s}$" % (kind, axis)


def plot_params(dfs: Sequence[pd.DataFrame], style):

    sns.set_style("whitegrid")

    default_df = [df for df in dfs if df["optimizer"][0] == EXPERT]
    assert len(default_df) > 0
    default_df = default_df[0]

    #fig, axs = plt.subplots(1, 2, figsize=(9, 2.5), constrained_layout=True, sharey=True)

    components = []
    styles = ["-", ":"]
    for df in dfs:
        if df["optimizer"][0] in [EXPERT, DETUNE]:
            continue
        if "trial" in df.columns and df["trial"][0] != 1:
            continue
        for axname in AXES:
            for gaintype in GAINTYPES:
                colname = f"{gaintype}_{axname}"
                th_fixedpoint = df[colname].to_numpy()
                th = np.exp(th_fixedpoint / (1 << 11))
                default_vals = default_df[colname].dropna().unique()
                assert len(default_vals) == 1
                default = np.exp(default_vals[0] / (1 << 11))
                ratio = th / default
                components.append(pd.DataFrame({
                    "optimizer": df["optimizer"],
                    "axis": axname,
                    "parameter": GAIN2DISPLAY[gaintype],
                    TIME: df[TIME],
                    RATIO_DEFAULT: ratio,
                }))
    df = pd.concat(components).reset_index()

    if style == BAD_INIT:
        grid = sns.relplot(
            df,
            kind="line",
            x=TIME,
            y=RATIO_DEFAULT,
            row="optimizer",
            row_order=[GAPS, EPISODIC, EPISODIC_STAR, SINGLEPOINT],
            col="axis",
            col_order=AXES,
            hue="parameter",
            hue_order=GAINTYPES_DISPLAY,
            height=1.5,
            aspect=1.75,
        )
        grid.set(xlim=[0, df[TIME].max()])
        grid.set_titles(template="{row_var}: {row_name}, {col_var}: {col_name}")
        for ax in grid.axes.flat:
            ax.axhline(1.0, color="black")
            ax.axhline(0.5, color="black", linestyle=":")
            if style != BAD_INIT:
                shade_fan(dfs[0], ax)
        sns.move_legend(
            grid,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncols=len(GAINTYPES),
        )

        grid.savefig(f"{style}_params.pdf")

    elif style == FAN:
        df = df.rename(columns=dict(parameter="param"))
        grid = sns.relplot(
            df,
            kind="line",
            x=TIME,
            y=RATIO_DEFAULT,
            col="axis",
            row="param",
            row_order=GAINTYPES_DISPLAY,
            hue="param",
            hue_order=GAINTYPES_DISPLAY,
            height=1.5,
            aspect=2.0,
            legend=False,
            facet_kws=dict(sharey=False),
        )
        grid.set(xlim=[0, df[TIME].max()], xticks=[0, 70, 140])
        grid.set_titles(template="{row_var}: {row_name}, {col_var}: {col_name}")
        for ax in grid.axes.flat:
            # zorder for main plot lines is > 1
            ax.axhline(1.0, color="black", zorder=1)
            ax.yaxis.set_label_coords(-0.2, 0.5)
            if style != BAD_INIT:
                handle = shade_fan(dfs[0], ax)

        grid.savefig(f"{style}_params.pdf")

    else:
        t0, t1 = dfs[0][TIME].min(), dfs[0][TIME].max()

        axs[-1].legend(
            frameon=False,
            title="param",
            loc="upper right",
            bbox_to_anchor=(1.015, 1.0),
            bbox_transform=fig.transFigure,
        )

        if style != BAD_INIT:
            for ax in axs:
                shade_fan(dfs[0], ax)
                ax.legend()

        fig.savefig(f"{style}_params.pdf")


def compare_params(dfs: Sequence[pd.DataFrame], style):
    thetas = list(it.product(AXES, GAINTYPES))

    # clip to shortest df
    tmax = min(df[TIME].max() for df in dfs)
    dfs = [df[df[TIME] <= tmax] for df in dfs]

    components = []
    for df in dfs:
        for ax, gaintype in thetas:
            colname = f"{gaintype}_{ax}"
            th_fixedpoint = df[colname].to_numpy()
            th = np.exp(th_fixedpoint / (1 << 11))
            ratio = th / th[df[colname].first_valid_index()]
            components.append(pd.DataFrame({
                "param": GAIN2DISPLAY[gaintype],
                "axis": ax,
                EXPERIMENT: df[EXPERIMENT][0],
                RATIO_DEFAULT: ratio,
                TIME: df[TIME],
            }))

    df = pd.concat(components).reset_index()

    sns.set_style("whitegrid")
    grid = sns.relplot(
        df,
        kind="line",
        col="axis",
        hue="param",
        hue_order=GAINTYPES_DISPLAY,
        row=EXPERIMENT,
        row_order=["weight", "fan"],
        x=TIME,
        y=RATIO_DEFAULT,
        height=2.0,
        aspect=1.1,
    )

    grid.set_titles(template=r"\textbf{{{row_var}:\! {row_name}}}; \; {col_var}:\! {col_name}")

    grid.set(xticks=np.linspace(0, 36, 4), xlim=[0, 36.05])
    yticks = [0.5, 1, 3, 10]
    ylabels = ["$1/2$"] + [f"${y}$" for y in [1, 3, 10]]
    grid.set(yscale="log", yticks=yticks, yticklabels=ylabels)
    for ax in grid.axes.flat:
        ax.yaxis.set_minor_locator(LogLocator(subs='all'))
        # Show minor ticks to emphasize log scale.
        ax.tick_params(axis="y", which="both", left=True, color="#CCC")

    # Make the legend one row on bottom.
    grid._legend.remove()
    grid.figure.legend(
        loc="lower center",
        frameon=False,
        title="parameter",
        ncol=len(grid._legend.get_texts()),
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.27) 

    grid.savefig(f"compare_params.pdf")


def episodic(df_gaps, dfs_episodic):
    def total_cost(df):
        df = df.bfill().ffill()
        summand = df["cost"][1:] * df[TIME].diff()[1:]
        return sum(summand)
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(figsize=(3.25, 2.0), constrained_layout=True)

    gapscost = total_cost(df_gaps)
    ax.axhline(gapscost, label="GAPS", color=GAPS_COLOR, linewidth=2)

    x = np.array([df["episode"][0] for df in dfs_episodic])
    y = np.array([total_cost(df) for df in dfs_episodic])
    ax.plot(
        x, y,
        label=EPISODIC,
        color=EPISODIC_COLOR,
        marker=".",
        markersize=12,
        markeredgewidth=0,
        markerfacecolor="black",
    )
    ax.set(xticks=[500, 1000, 1500, 2000, 2500, 3000])
    ax.set(yticks=[0.25, 0.275, 0.3, 0.325], ylim=[0.245, 0.335])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.minorticks_off()
    ax.legend()
    ax.set(xlabel="episode length", ylabel="total cost")

    extra_pct = 100 * (y / gapscost - 1)
    best, worst = np.amin(extra_pct), np.amax(extra_pct)
    print(f"Episodic: cost ratio ranges between {best} and {worst}.")

    sns.despine(ax=ax, left=True, bottom=True)
    fig.savefig("episodic.pdf")
    fig.savefig("episodic.png", dpi=200)


def main():
    style = sys.argv[-1]
    assert style in STYLES

    replace = {
        "default": EXPERT,
        "GAPS": GAPS,
        "episodic": EPISODIC,
        r"episodic$\star$": EPISODIC_STAR,
        "singlepoint": SINGLEPOINT,
    }

    paths = sys.argv[1:-1]
    dfs = []
    for path in paths:
        df = pd.read_json(path)
        opts = df["optimizer"].unique()
        for k, v in replace.items():
            if k in opts:
                df["optimizer"] = df["optimizer"].str.replace(k, v)
        df[TIME] = df["t"] - df["t"][0]
        dfi = df.interpolate()
        cost = sum((dfi[f"target_{c}"] - dfi[f"pos_{c}"]) ** 2 for c in "xyz")
        dfi["cost"] = cost
        dfi[ERR] = np.sqrt(cost) * 100
        experiment = Path(path).stem.split("_")[0]
        dfi[EXPERIMENT] = experiment
        dfs.append(dfi)

    if True:
        plt.rcParams.update({"text.usetex": True, "font.size": 12})

    if style == MULTI_PARAM:
        compare_params(dfs, style)
    elif style == EPISODIC:
        episodic(dfs[0], dfs[1:])
    else:
        plot_params(dfs, style)
        plot_costs(dfs, style)
        if style == BAD_INIT:
            plot_fig8(dfs, style)


if __name__ == "__main__":
    main()

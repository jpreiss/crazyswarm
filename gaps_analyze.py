from collections import defaultdict
import datetime
from pathlib import Path
import sys
import itertools as it
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import patheffects
import numpy as np
import pandas as pd
import seaborn as sns

# styles
BASIC = "basic"
BAD_INIT = "bad_init"
FAN = "fan"
WEIGHT = "weight"
MULTI_PARAM = "multi_param"
STYLES = [BASIC, BAD_INIT, FAN, WEIGHT, MULTI_PARAM]

# column names
TIME = "time (sec)"
ERR = "tracking error (cm)"
COST_CUM = "cumulative cost"
REGRET = "``regret'' vs. default"
EXPERIMENT = "experiment"
LOG_RATIO_INIT = r"$\log_2(\mathrm{value} / \mathrm{initial})$"
LOG_RATIO_DEFAULT = r"$\log_2$(value / default)"


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
        handle = ax.axvspan(*pair, alpha=0.15, color="black", linewidth=0, label=label)
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

    width = len(dfs) * (1.5 if style == FAN else 3.5)
    fig_fig8, axs_fig8 = plt.subplots(
        1, len(dfs),
        figsize=(width, 2.7),
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
        df = df[keep_cols].copy()
        df[TIME] = pd.to_timedelta(df[TIME], unit="seconds")
        df = df.set_index(TIME)
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
                shift = -np.array([0.25, 0.9])
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

    cbar = fig_fig8.colorbar(line)
    cbar.ax.set_ylabel(TIME)
    fig_fig8.savefig(f"{style}_fig8.pdf")


def plot_costs(dfs: Sequence[pd.DataFrame], style):

    # TODO: work on the layout of this figure + what should stay in final paper.
    fig_cost, axs_cost = plt.subplots(4, 1, figsize=(8, 8), constrained_layout=True)
    ax_cost, ax_err, ax_cum, ax_regret = axs_cost

    # take downsampled means to smoot the plots a little.
    maxtime = max(df[TIME].max() for df in dfs)
    dfs_sampled = []
    for df in dfs:
        df["timedelta"] = pd.to_timedelta(df[TIME], unit="seconds")
        dfi = df.set_index("timedelta")
        dfr = dfi.resample("100ms").apply(agg)
        # this used to be before resampling, but that was wrong!
        dfr[COST_CUM] = (dfr["cost"] * dfr[TIME].diff()).cumsum()
        dfs_sampled.append(dfr)
    dfs = dfs_sampled

    # TODO: figure out a more SQL-y way to do this. Ideally we wouldn't even
    # need the dataframe split.
    df_base = [df for df in dfs if df["optimizer"].iloc[0] == "default"]
    assert len(df_base) == 1
    df_base = df_base[0]
    for df in dfs:
        df[REGRET] = df[COST_CUM] - df_base[COST_CUM]
    dfcat = pd.concat(dfs).reset_index()

    sns.lineplot(dfcat, ax=ax_cost, x=TIME, y="cost", hue="optimizer")
    sns.lineplot(dfcat, ax=ax_err, x=TIME, y=ERR, hue="optimizer")
    sns.lineplot(dfcat, ax=ax_cum, x=TIME, y=COST_CUM, hue="optimizer")
    sns.lineplot(dfcat, ax=ax_regret, x=TIME, y=REGRET, hue="optimizer")
    # TODO: work on legends.

    if style != BAD_INIT:
        for ax in axs_cost:
            shade_fan(dfs[0], ax)
            ax.legend()

    fig_cost.savefig(f"{style}_cost.pdf")


def plot_costs_v2(dfs: Sequence[pd.DataFrame], style):

    sns.set_style("whitegrid")

    fig_cost, axs = plt.subplots(1, 2, figsize=(10, 2.25), constrained_layout=True)
    ax_err, ax_regret = axs

    # take downsampled means to smooth the plots a little.
    maxtime = max(df[TIME].max() for df in dfs)
    dfs_sampled = []
    for df in dfs:
        df["timedelta"] = pd.to_timedelta(df[TIME], unit="seconds")
        dfi = df.set_index("timedelta")
        dfr = dfi.resample("100ms").apply(agg)
        # this used to be before resampling, but that was wrong!
        dfr[COST_CUM] = (dfr["cost"] * dfr[TIME].diff()).cumsum()
        dfs_sampled.append(dfr)
    dfs = dfs_sampled

    # TODO: figure out a more SQL-y way to do this. Ideally we wouldn't even
    # need the dataframe split.
    df_base = [df for df in dfs if df["optimizer"].iloc[0] == "none"]
    assert len(df_base) == 1
    df_base = df_base[0]
    for df in dfs:
        df[REGRET] = df[COST_CUM] - df_base[COST_CUM]
    dfcat = pd.concat(dfs).reset_index()

    style_order = ["gaps", "singlepoint", "ogd", "detune", "none"]
    kwargs = dict(
        data=dfcat,
        x=TIME,
        color="black",
        style="optimizer",
        style_order=style_order,
        size="optimizer",
        #sizes=defaultdict(lambda: 2, none=1.0, detune=1.0),
    )

    sns.lineplot(ax=ax_err, y=ERR, legend=False, **kwargs)
    sns.lineplot(ax=ax_regret, y=REGRET, **kwargs)
    sns.move_legend(ax_regret, "upper left", bbox_to_anchor=(1, 1), frameon=False)

    _, emax = ax_err.get_ylim()
    ax_err.set_ylim([0, emax])

    ax_regret.set_ylim([-0.03, 0.6])

    if style == BAD_INIT:
        for ax in axs:
            ax.set(xticks=np.linspace(0, 32, 5), xlim=(0, 32))
            # put GAPS in front
            ax.lines[0].set(zorder=100)

    if style != BAD_INIT:
        for ax in axs:
            shade_fan(dfs[0], ax)
            ax.legend()

    fig_cost.savefig(f"{style}_cost.pdf")


def param_format(p):
    """Converts code-style names for controller parameters to LaTeX."""
    if p[0] != "k" or p[2] != "_":
        return p
    kind = p[1]
    axis = p[3:]
    return "$k_{%s}^{%s}$" % (kind, axis)


def plot_params(dfs: Sequence[pd.DataFrame], style):

    sns.set_style("whitegrid")

    default_df = [df for df in dfs if df["optimizer"][0] == "none"]
    assert len(default_df) == 1
    default_df = default_df[0]

    #fig, axs = plt.subplots(1, 2, figsize=(9, 2.5), constrained_layout=True, sharey=True)

    components = []
    styles = ["-", ":"]
    for df in dfs:
        #for ax, axname in zip(axs, ["xy", "z"]):
        for axname in ["xy", "z"]:
            #ax.set_title(f"axis: {axname}", fontsize=12)
            for gaintype in ["ki", "kp", "kv", "kr", "kw"]:
                colname = f"{gaintype}_{axname}"
                th_fixedpoint = df[colname].to_numpy()
                th = np.exp(th_fixedpoint / (1 << 11))
                default_vals = default_df[colname].dropna().unique()
                assert len(default_vals) == 1
                default = np.exp(default_vals[0] / (1 << 11))
                ratio = th / default
                #ax.plot(df[TIME], np.log2(ratio), label=gaintype)
                components.append(pd.DataFrame({
                    "optimizer": df["optimizer"],
                    "axis": axname,
                    "parameter": gaintype,
                    TIME: df[TIME],
                    LOG_RATIO_DEFAULT: ratio,
                }))
    df = pd.concat(components).reset_index().sample(frac=0.01)

    if True:
        grid = sns.relplot(
            df,
            kind="line",
            x=TIME,
            y=LOG_RATIO_DEFAULT,
            style="optimizer",
            col="axis",
            hue="parameter",
        )
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
    gaintypes = ["ki", "kp", "kv", "kr", "kw"]
    axes = ["xy", "z"]
    thetas = list(it.product(axes, gaintypes))
    thetas_pretty = [param_format(p) for p in thetas]
    runtype = EXPERIMENT if style == MULTI_PARAM else "optimizer"

    # clip to shortest df
    tmax = min(df[TIME].max() for df in dfs)
    dfs = [df[df[TIME] <= tmax] for df in dfs]

    components = []
    for df in dfs:
        for ax, gaintype in thetas:
            colname = f"{gaintype}_{ax}"
            #df[theta] = df[theta] - df[theta].first()
            th_fixedpoint = df[colname].to_numpy()
            th = np.exp(th_fixedpoint / (1 << 11))
            ratio = th / th[df[colname].first_valid_index()]
            components.append(pd.DataFrame({
                "param": gaintype,
                "axis": ax,
                runtype: df[runtype][0],
                LOG_RATIO_INIT: np.log2(ratio),
                TIME: df[TIME],
            }))
            print(components[-1])

    df = pd.concat(components).reset_index()

    sns.set_style("whitegrid")
    grid = sns.relplot(
        df,
        kind="line",
        col="axis",
        hue="param",
        row=runtype,
        row_order=["weight", "fan"],
        #col_order=thetas_pretty,
        #col_order=thetas_pretty,
        #col_wrap=5,
        x=TIME,
        #y="value",
        y=LOG_RATIO_INIT,
        height=2.5,
        aspect=1.75,
    )

    if style != BAD_INIT:
        for ax in grid.axes[1]:
            handle = shade_fan(dfs[1], ax)  # index is a hack - should inspect df

    grid.set_titles(template=r"\textbf{{{row_var}:\! {row_name}}}; \; {col_var}:\! {col_name}")
    grid.set(xticks=np.linspace(0, 36, 7), xlim=[0, 36.05])
    grid.set(yticks=[-0.5, 0, 0.5, 1.0, 1.5, 2.0], ylim=[-0.5, 2.01])
    sns.move_legend(
        grid,
        loc="right",
        bbox_to_anchor=(1.03, 0.6),
        bbox_transform=grid.figure.transFigure
    )

    grid.axes[1, 1].legend(
        title="fan state",
        handles=[handle], labels=["on"],
        frameon=False,
        bbox_to_anchor=(1.03, 0.35),
        bbox_transform=grid.figure.transFigure
    )

    grid.savefig(f"compare_params.pdf")


def main():
    style = sys.argv[-1]
    assert style in STYLES

    paths = sys.argv[1:-1]
    dfs = []
    for path in paths:
        df = pd.read_json(path)
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
    else:
        plot_params(dfs, style)
        plot_fig8(dfs, style)
        plot_costs_v2(dfs, style)


if __name__ == "__main__":
    main()

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
LOG_RATIO = r"$\log_2(\mathrm{value} / \mathrm{initial})$"


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
    width = len(dfs) * (1.5 if style == FAN else 3.5)
    fig_fig8, axs_fig8 = plt.subplots(
        1, len(dfs),
        figsize=(width, 2.4),
        constrained_layout=True,
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

        target = np.stack([df[c] for c in target_cols], axis=1)
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
                pos = pos[:, [0, -1]]
                target = target[:, [0, -1]]

        ax.plot(*target.T, label="target", color="gray", linewidth=1)

        cmap = "coolwarm"
        line = plot_colorchanging(
            ax, *pos.T, maxtime=maxtime,
            label=name, cmap=cmap, linewidth=2,
        )
        ax.set(title=name, xlabel="horizontal (m)", ylabel="vertical (m)")
        ax.axis("equal")

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


def param_format(p):
    """Converts code-style names for controller parameters to LaTeX."""
    if p[0] != "k" or p[2] != "_":
        return p
    kind = p[1]
    axis = p[3:]
    return "$k_{%s}^{%s}$" % (kind, axis)


def plot_params(dfs: Sequence[pd.DataFrame], style):
    gaintypes = ["ki", "kp", "kv", "kr", "kw"]
    axes = ["xy", "z"]
    thetas = [f"{p}_{s}" for s, p in it.product(axes, gaintypes)]
    thetas_pretty = [param_format(p) for p in thetas]

    df = pd.concat(dfs)
    df = df.rename(mapper=param_format, axis="columns")

    cols = [k for k, v in df.items() if k in thetas_pretty]

    df = df.melt(id_vars=["optimizer", TIME], value_vars=cols, var_name="param")
    df["value"] = np.exp(df["value"].to_numpy() / (1 << 12))

    grid = sns.relplot(
        df,
        kind="line",
        hue="optimizer",
        col="param",
        col_wrap=4,
        x=TIME,
        y="value",
        height=1.75,
        aspect=1.5,
        facet_kws=dict(
            sharey=False,
        )
    )
    sns.move_legend(
        grid,
        loc="lower center",
        bbox_to_anchor=(0.48, -0.1),
        ncols=2,
    )
    if style != BAD_INIT:
        for ax in grid.axes.flat:
            shade_fan(dfs[0], ax)
            ax.legend()
    grid.savefig(f"{style}_params.pdf")


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
            th = np.exp(th_fixedpoint / (1 << 12))
            ratio = th / th[df[colname].first_valid_index()]
            components.append(pd.DataFrame({
                "param": gaintype,
                "axis": ax,
                runtype: df[runtype][0],
                LOG_RATIO: np.log2(ratio),
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
        y=LOG_RATIO,
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
    style = sys.argv[1]
    assert style in STYLES

    paths = sys.argv[2:]
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
        plot_costs(dfs, style)


if __name__ == "__main__":
    main()

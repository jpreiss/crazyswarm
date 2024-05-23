import sys
import itertools as it
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
import seaborn as sns

# styles
BASIC = "basic"
BAD_INIT = "bad_init"
STYLES = [BASIC, BAD_INIT]

# column names
TIME = "$t$"
ERR = "tracking error (cm)"


def agg(series):
    if series.dtype == "object":
        return series.iloc[0]
    return series.mean()


def plot_colorchanging(ax, x, y, *args, **kwargs):
    # Create a set of line segments so that we can color them individually
    # This creates the points as an N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, *args, **kwargs)
    # Set the values used for colormapping
    lc.set_array(np.linspace(0, 1, len(x)))
    ax.add_collection(lc)
    return lc
    #fig.colorbar(line, ax=axs[0])


def shade_fan(df, ax):
    fan_state = df["fan"] == 1
    fan_state = fan_state.bfill().ffill()
    fan_state.iloc[-1] = False
    fan_toggles = np.flatnonzero(np.diff(fan_state) != 0)
    assert len(fan_toggles) & 0x1 == 0
    # convert from index to time
    fan_toggles = df[TIME].iloc[fan_toggles].to_numpy()
    label = "fan on"
    for pair in fan_toggles.reshape((-1, 2)):
        ax.axvspan(*pair, alpha=0.2, color="black", linewidth=0, label=label)
        label = None


def plot_fig8(dfs, style):
    width = len(dfs) * 3.5
    fig_fig8, axs_fig8 = plt.subplots(
        1, len(dfs),
        figsize=(width, 2.4),
        constrained_layout=True,
    )
    for ax, df in zip(axs_fig8, dfs):
        name = df["optimizer"].iloc[0]

        ax.plot(df["target_x"], df["target_z"], label="target", color="gray", linewidth=1)

        # The 10ms interp is a bit slow, so only grab the columns we need.
        df = df[[TIME, "target_x", "target_z", "pos_x", "pos_z"]].copy()
        df[TIME] = pd.to_timedelta(df[TIME], unit="seconds")
        df = df.set_index(TIME)
        df = df.resample("10ms").apply(agg)

        cmap = "viridis"
        line = plot_colorchanging(ax, df["pos_x"], df["pos_z"], label=name, cmap=cmap, linewidth=2)
        ax.set(title=name, xlabel="horizontal (m)", ylabel="vertical (m)")
        ax.axis("equal")

    cbar = fig_fig8.colorbar(line)
    cbar.ax.set_ylabel("time")
    fig_fig8.savefig(f"{style}_fig8.pdf")


def plot_costs(dfs: Sequence[pd.DataFrame], style):

    fig_cost, axs_cost = plt.subplots(4, 1, figsize=(8, 8), constrained_layout=True)
    ax_cost, ax_err, ax_cum, ax_regret = axs_cost

    # interpolate all DFs at the union of all time values.
    maxtime = max(df[TIME].max() for df in dfs)

    interps = []
    for df in dfs:
        df[TIME] = pd.to_timedelta(df[TIME], unit="seconds")
        dfi = df.set_index(TIME)
        dfr = dfi.resample("100ms").apply(agg)
        interps.append(dfr)

    dfs = interps

    #for df in dfs[1:]:
        #assert len(df) == len(dfs[0])

    # TODO: figure out a more SQL-y way to do this. Ideally we wouldn't even
    # need the dataframe split.
    df_base = [df for df in dfs if df["optimizer"].iloc[0] == "default"]
    assert len(df_base) == 1
    df_base = df_base[0]
    for df in dfs:
        df["regret"] = df["cost_cum"] - df_base["cost_cum"]
    dfcat = pd.concat(dfs).reset_index()

    sns.lineplot(dfcat, ax=ax_cost, x=TIME, y="cost", hue="optimizer")
    sns.lineplot(dfcat, ax=ax_err, x=TIME, y=ERR, hue="optimizer")
    sns.lineplot(dfcat, ax=ax_cum, x=TIME, y="cost_cum", hue="optimizer")
    sns.lineplot(dfcat, ax=ax_regret, x=TIME, y="regret", hue="optimizer")

    if style != BAD_INIT:
        for ax in axs_cost:
            shade_fan(dfs[0], ax)
            ax.legend()

    fig_cost.savefig(f"{style}_cost.pdf")


def param_format(p):
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

    dfs = [df.sample(frac=0.1) for df in dfs]
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
    grid.set(xticks=[0, 16, 32])
    if style != BAD_INIT:
        for ax in grid.axes.flat:
            shade_fan(dfs[0], ax)
            ax.legend()
    grid.savefig(f"{style}_params.pdf")


def main():
    style = sys.argv[1] if len(sys.argv) > 1 else BASIC
    assert style in STYLES

    dfs = []
    for mode in ["true_false", "false_false"]:
        df = pd.read_json(f"/home/james/.ros/gaps_{mode}.json")
        df[TIME] = df["t"] - df["t"][0]
        dfi = df.interpolate()
        cost = sum((dfi[f"target_{c}"] - dfi[f"pos_{c}"]) ** 2 for c in "xyz")
        df["cost"] = cost
        df[ERR] = np.sqrt(cost) * 100
        df["cost_cum"] = cost.cumsum()
        dfs.append(df)

    if False:
        plt.rcParams.update({"text.usetex": True, "font.size": 12})

    plot_fig8(dfs, style)
    #plot_costs(dfs, style)
    #plot_params(dfs, style)


if __name__ == "__main__":
    main()

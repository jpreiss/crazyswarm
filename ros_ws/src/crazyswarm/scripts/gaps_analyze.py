import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


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


def main(adapt=False):
    W = 8
    H = 1.5
    def subplots(n):
        return plt.subplots(n, 1, figsize=(W, H*n), constrained_layout=True)

    z_default = np.load("default.npz")
    z_gaps = np.load("gaps.npz")
    names = ["default", "GAPS"]

    state_logs = []
    targets = []
    cost_logs = []

    prefix = "basic"

    for npz in [z_gaps]:
        pos = npz["state"]
        target = npz["target"]
        cost = np.sum((pos - target) ** 2, axis=1)
        state_logs.append(pos)
        targets.append(target)
        cost_logs.append(cost)

    fig_fig8, axs_fig8 = plt.subplots(1, 2, figsize=(8.5, 2.5), constrained_layout=True)
    for ax, pos, target, name in zip(axs_fig8, state_logs, targets, names):
        ax.plot(target[:, 0], target[:, 2], label="target", color="gray", linewidth=1)
        cmap = "viridis"
        line = plot_colorchanging(ax, pos[:, 0], pos[:, 2], label=name, cmap=cmap, linewidth=2)
        ax.set(title=name, xlabel="horizontal (m)", ylabel="gravity (m)")
        ax.axis("equal")
    cbar = fig_fig8.colorbar(line)
    cbar.ax.set_ylabel("time")
    fig_fig8.savefig(f"{prefix}_fig8.pdf")
    return

    fig_cost, axs_cost = subplots(3)
    ax_cost, ax_cum, ax_regret = axs_cost
    for log, name in zip(cost_logs, names):
        t = np.arange(len(log)) / 50 # TODO: log time
        ax_cost.plot(t, log, label=name)
        ax_cum.plot(t, np.cumsum(log), label=name)
    ax_regret.plot(t, np.cumsum(cost_logs[1] - cost_logs[0]))
    ax_cost.set(ylabel="cost")
    ax_cum.set(ylabel="cumulative cost")
    ax_regret.set(ylabel="cum. cost difference")
    for ax in axs_cost:
        ax.legend()
    fig_cost.savefig(f"{prefix}_cost.pdf")


if __name__ == "__main__":
    main()

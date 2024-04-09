from copy import copy, deepcopy
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import cffirmware
from crazyflie_sim.crazyflie_sil import CrazyflieSIL
from crazyflie_sim.backend.np import Quadrotor
from crazyflie_sim.sim_data_types import State


def norm2(x):
    return np.sum(x ** 2)


PARAM_ATTRS = [
    p + s for p, s in it.product(["kp_", "kd_", "ki_"], ["xy", "z"])
]
HZ = 500


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


def rollout(sim: Quadrotor, cf: CrazyflieSIL, adapt: bool):

    ticks = 0

    radius = 1.0
    period = 6.0
    omega = 2 * np.pi / period
    init_pos = cf.initialPosition

    repeats = 4 + 4 * adapt
    T = int(repeats * period * HZ) + 1

    rng = np.random.default_rng(0)
    w = 1e-2 * rng.normal(size=(T, 3))

    sim.state.pos = init_pos
    sim.state.vel[2] = radius * omega

    state_log = []
    target_log = []
    cost_log = []
    param_log = []
    action_log = []
    y_log = []

    # setpoint
    init_pos_vec = cffirmware.mkvec(*init_pos)
    pos = init_pos
    vel = np.zeros(3)
    acc = np.zeros(3)
    yaw = 0
    angvel = np.zeros(3)

    # introduce a constant "wind" pushing us out of the desired fig8-plane
    if adapt:
        wind_osc = np.cos(2 * (2 * np.pi / T) * np.arange(T) - 0.4 * np.pi)
        wind_y = np.clip(5 * wind_osc, -1, 1) - 1
        w[:, 1] += 0.05 * wind_y
    #plt.plot(w[:, 1])
    #plt.show()


    for t in range(T):

        tsec = ticks / (2 * HZ)
        pos[0] = radius * np.cos(omega * tsec) - radius
        pos[2] = radius * 0.5 * np.sin(2 * omega * tsec)
        pos += init_pos_vec
        vel[0] = -radius * omega * np.sin(omega * tsec)
        vel[2] = radius * 1 * omega * np.cos(2 * omega * tsec)
        acc[0] = -radius * (omega ** 2) * np.cos(omega * tsec)
        acc[2] = -radius * 2 * (omega ** 2) * np.sin(2 * omega * tsec)
        state_log.append(deepcopy(sim.state))
        target_log.append(State(pos=pos, vel=vel))
        assert cf.mellinger_control.gaps_Qv == 0.0
        cost_log.append(
            0.5 * cf.mellinger_control.gaps_Qx * norm2(pos - sim.state.pos)
        )
        cf.setState(sim.state)
        cf.cmdFullState(pos, vel, acc, yaw, angvel)

        theta = [getattr(cf.mellinger_control.gaps, k) for k in PARAM_ATTRS]
        param_log.append(theta)
        y_log.append(cf.mellinger_control.gaps.y)

        action = cf.executeController()
        action_arr = list(action.rpm) + [
            cf.mellinger_control.cmd_roll,
            cf.mellinger_control.cmd_pitch,
            cf.mellinger_control.cmd_yaw,
            cf.mellinger_control.cmd_thrust,
        ]
        action_log.append(action_arr)

        # Mellinger expects to run at 500 Hz.
        action2 = cf.executeController()
        assert action2.rpm == action.rpm
        ticks += 2

        f_disturb = w[t]
        sim.step(action, 1.0 / HZ, f_disturb)

    return state_log, target_log, cost_log, param_log, action_log, y_log


def main(adapt: bool):
    cfs = [
        CrazyflieSIL("", np.array([0.0, 0.0, 1.0]), "mellinger")
        for _ in range(2)
    ]
    for cf in cfs:
        # without this, the simulation is unstable
        cf.mellinger_control.kd_omega_rp = 0
        cf.mellinger_control.mass = Quadrotor(State()).mass
        # because it's annoying to extract the "u"
        cf.mellinger_control.gaps_Qv = 0
        cf.mellinger_control.gaps_R = 0
        cf.mellinger_control.i_range_xy *= 0.5

    cfs[1].mellinger_control.gaps_enable = True
    cfs[1].mellinger_control.gaps_eta = 1e-2

    results = [
        rollout(Quadrotor(State()), cf, adapt=adapt)
        for cf in cfs
    ]

    state_logs = [results[0][0], results[1][0], results[1][1]]
    cost_logs = [results[0][2], results[1][2]]
    cost_logs = [np.array(a) for a in cost_logs]
    names = ["default", "GAPS", "target"]

    t = np.arange(len(state_logs[0])) / HZ

    target = np.stack([step.pos for step in results[1][1]])
    fig, ax = plt.subplots()
    ax.plot(target[:, 0], target[:, 2], color="black")
    ax.axis("equal")
    ax.set(ylabel="gravity", xlabel="horizontal")
    fig.savefig("target.pdf")


    W = 8
    H = 1.5
    def subplots(n):
        return plt.subplots(n, 1, figsize=(W, H*n), constrained_layout=True)

    prefix = "gaps_cf"
    if adapt:
        prefix += "_adapt"

    fig_fig8, axs_fig8 = plt.subplots(1, 2, figsize=(8.5, 2.5), constrained_layout=True)
    for ax, log, name in zip(axs_fig8, state_logs[:2], names[:2]):
        pos = np.stack([s.pos for s in log])
        ax.plot(target[:, 0], target[:, 2], label="target", color="gray", linewidth=1)
        #ax.plot(pos[:, 0], pos[:, 2], label=name, color="black", linewidth=2)
        cmap = "viridis"
        line = plot_colorchanging(ax, pos[:, 0], pos[:, 2], label=name, cmap=cmap, linewidth=2)
        ax.set(title=name, xlabel="horizontal (m)", ylabel="gravity (m)")
        ax.axis("equal")
    cbar = fig_fig8.colorbar(line)
    cbar.ax.set_ylabel("time")
    fig_fig8.savefig(f"{prefix}_fig8.pdf")

    fig_pos, axs_pos = subplots(2)
    fig_vel, axs_vel = subplots(2)
    for log, name in zip(state_logs, names):
        for ax_p, ax_v, coord in zip(axs_pos, axs_vel, [0, 2]):
            ax_p.plot(t, [s.pos[coord] for s in log], label=name)
            ax_v.plot(t, [s.vel[coord] for s in log], label=name)
            for ax in [ax_p, ax_v]:
                ax.set(ylabel=["x", "y", "z"][coord])
    for ax in axs_pos:
        ax.legend()
    for ax in axs_vel:
        ax.legend()
    fig_pos.savefig(f"{prefix}_pos.pdf")
    fig_vel.savefig(f"{prefix}_vel.pdf")

    fig_cost, axs_cost = subplots(3)
    ax_cost, ax_cum, ax_regret = axs_cost
    for log, name in zip(cost_logs, names):
        ax_cost.plot(t, log, label=name)
        ax_cum.plot(t, np.cumsum(log), label=name)
    ax_regret.plot(t, np.cumsum(cost_logs[1] - cost_logs[0]))
    ax_cost.set(ylabel="cost")
    ax_cum.set(ylabel="cumulative cost")
    ax_regret.set(ylabel="cum. cost difference")
    for ax in axs_cost:
        ax.legend()
    fig_cost.savefig(f"{prefix}_cost.pdf")

    param_logs = np.stack(results[1][3])
    T, theta_dim = param_logs.shape
    fig_param, axs_param = subplots(theta_dim)
    for trace, ax, name in zip(param_logs.T, axs_param, PARAM_ATTRS):
        ax.plot(t, trace)
        ax.set_ylabel(name)
    fig_param.savefig(f"{prefix}_params.pdf")

    action_logs = [np.stack(r[4]) for r in results]
    T, ac_dim = action_logs[0].shape
    fig_action, axs_action = subplots(ac_dim)
    for log, name in zip(action_logs, names):
        for trace, ax in zip(log.T, axs_action):
            ax.plot(t, trace, label=name)
    for ax in axs_action:
        ax.legend()
    fig_action.savefig(f"{prefix}_actions.pdf")

    y_log = np.stack(results[1][5])
    assert y_log[0].shape == (9, 6)
    fig_y, ax_y = subplots(1)
    maxes = [np.amax(y.flat) for y in y_log]
    ax_y.plot(t, maxes)
    fig_y.savefig(f"{prefix}_ymax.pdf")

if __name__ == "__main__":
    main(adapt=False)
    main(adapt=True)

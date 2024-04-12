from pathlib import Path
from copy import copy, deepcopy
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from pycrazyswarm import *


def norm2(x):
    return np.sum(x ** 2)


PARAM_ATTRS = [
    p + s for p, s in it.product(["kp_", "kd_", "ki_"], ["xy", "z"])
]
HZ = 500


class RampTime:
    def __init__(self, ramptime, faketime):
        # see ramptime.py for symbolic derivation
        T = ramptime
        X = faketime
        assert X <= T
        self.a = 1 / T**2 - 2 * X / T**3
        self.b = 3 * X / T**2 - 1 / T
        self.T = T
        self.X = X
        # test
        Tbefore = np.nextafter(T, 0)
        Tafter = np.nextafter(T, T + 1)
        assert np.isclose(self.val(0), 0)
        assert np.isclose(self.deriv(0), 0)
        assert np.isclose(self.val(Tbefore), X)
        assert np.isclose(self.deriv(Tbefore), 1)
        assert np.isclose(self.val(Tafter), X)
        assert np.isclose(self.deriv(Tafter), 1)
        assert np.isclose(self.val(T + 1), X + 1)
        assert np.isclose(self.deriv(T + 1), 1)
        assert np.isclose(self.val(T + 2), X + 2)
        assert np.isclose(self.deriv(T + 2), 1)

    def val(self, t):
        if t < self.T:
            return self.a * t**3 + self.b * t**2
        return (t - self.T) + self.X

    def deriv(self, t):
        if t < self.T:
            return 3 * self.a * t**2 + 2 * self.b * t
        return 1.0


def rollout2(cf, Z, timeHelper):
    radius = 0.75
    period = 4
    omega = 2 * np.pi / period
    init_pos = cf.initialPosition + [0, 0, Z]
    assert Z > radius / 2 + 0.2
    print("init_pos is", init_pos)

    repeats = 10

    state_log = []
    target_log = []
    cost_log = []
    param_log = []
    action_log = []
    y_log = []

    # setpoint
    pos = init_pos.copy()
    vel = np.zeros(3)
    acc = np.zeros(3)
    yaw = 0
    angvel = np.zeros(3)

    # time ramp
    t0 = timeHelper.time()
    ramp = RampTime(1.5 * period, period)
    rampdown_begin = None

    while True:
        ttrue = timeHelper.time() - t0
        tramp = ramp.val(ttrue)
        if tramp > (repeats + 1) * period and rampdown_begin is None:
            rampdown_begin = tramp
        if rampdown_begin is None:
            tsec = tramp
            tderiv = ramp.deriv(ttrue)
        else:
            argt = ramp.T - (tramp - rampdown_begin)
            assert argt >= -1/50
            assert argt <= ramp.T
            tsec = rampdown_begin + ramp.X - ramp.val(argt)
            tderiv = ramp.deriv(argt)

            if tsec > (repeats + 2) * period:
                break

        pos[0] = radius * np.cos(omega * tsec) - radius
        pos[2] = radius * 0.5 * np.sin(2 * omega * tsec)
        pos[1] = -pos[2]
        pos += init_pos
        #print(f"{pos = }")
        #print(f"{init_pos = }")
        omega2 = tderiv * omega
        vel[0] = -radius * omega2 * np.sin(omega * tsec)
        vel[2] = radius * 1 * omega2 * np.cos(2 * omega * tsec)
        vel[1] = -vel[2]
        acc[0] = -radius * (omega2 ** 2) * np.cos(omega * tsec)
        acc[2] = -radius * 2 * (omega2 ** 2) * np.sin(2 * omega * tsec)
        acc[1] = -acc[2]

        if tsec > period and rampdown_begin is None:
            state_log.append(cf.position())
            target_log.append(pos.copy())
            #cost_log.append(
                #0.5 * cf.mellinger_control.gaps_Qx * norm2(pos - sim.state.pos)
            #)
        cf.cmdFullState(pos, vel, acc, yaw, angvel)

        #theta = [getattr(cf.mellinger_control.gaps, k) for k in PARAM_ATTRS]
        #param_log.append(theta)
        #y_log.append(cf.mellinger_control.gaps.y)

        timeHelper.sleepForRate(50)

    return state_log, target_log #, cost_log, param_log, action_log, y_log


def main2(gaps: bool):
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Z = 0.7

    # params
    GAPS_Qv = 0.0
    GAPS_R = 0.0
    GAPS_ETA = 1e-2
    GAPS_DAMPING = 0.9995

    if gaps:
        cf.setParams({
            "gaps/Qv": GAPS_Qv,
            "gaps/R": GAPS_R,
            "gaps/eta": GAPS_ETA,
            "gaps/damping": GAPS_DAMPING,
        })
        cf.setParam("gaps/enable", 1)


    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=2.0)
    timeHelper.sleep(3.0)

    state_log, target_log = rollout2(cf, Z, timeHelper)

    cf.notifySetpointsStop()
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=2.0)
    timeHelper.sleep(3.0)

    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    name = "gaps.npz" if gaps else "default.npz"
    np.savez(name, state=state_log, target=target_log)


def main_old(adapt: bool):
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


def main():
    main2(gaps=False)


if __name__ == "__main__":
    main()

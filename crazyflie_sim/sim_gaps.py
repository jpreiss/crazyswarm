from copy import copy, deepcopy
import itertools as it

import matplotlib.pyplot as plt
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


def rollout(sim: Quadrotor, cf: CrazyflieSIL):
    HZ = 1000

    ticks = 0
    def time_func():
        nonlocal ticks
        return ticks / float(HZ)
    # HACK!!!
    cf.time_func = time_func

    radius = 0.7
    period = 5.0
    omega = 2 * np.pi / period

    repeats = 8
    T = int(repeats * period * HZ / 2) + 1

    state_log = []
    target_log = []
    cost_log = []
    param_log = []

    # setpoint
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    yaw = 0
    angvel = np.zeros(3)

    for t in range(T):
        tsec = time_func()
        pos[0] = radius * np.cos(omega * tsec) - radius
        pos[2] = radius * np.sin(omega * tsec)
        vel[0] = -radius * omega * np.sin(omega * tsec)
        vel[2] = radius * omega * np.cos(omega * tsec)
        acc[0] = -radius * (omega ** 2) * np.cos(omega * tsec)
        acc[2] = -radius * (omega ** 2) * np.sin(omega * tsec)
        state_log.append(deepcopy(sim.state))
        target_log.append(State(pos=pos, vel=vel))
        # TODO: control cost!!!
        cost_log.append(
            0.5 * cf.mellinger_control.gaps_Qx * norm2(pos - sim.state.pos)
            + 0.5 * cf.mellinger_control.gaps_Qv * norm2(vel - sim.state.vel)
        )
        cf.setState(sim.state)
        cf.cmdFullState(pos, vel, acc, yaw, angvel)

        theta = [getattr(cf.mellinger_control.gaps, k) for k in PARAM_ATTRS]
        param_log.append(theta)

        # Mellinger expects to run at 500 Hz.
        action = cf.executeController()
        ticks += 1
        cf.executeController()
        ticks += 1

        f_disturb = np.zeros(3)
        sim.step(action, 2.0 / HZ, f_disturb)

    print()

    return state_log, target_log, cost_log, param_log


def main():
    HZ = 1000

    cfs = [
        CrazyflieSIL("", np.zeros(3), "mellinger", lambda: 0)
        for _ in range(2)
    ]
    for cf in cfs:
        cf.mellinger_control.kd_xy *= 8
        cf.mellinger_control.gaps.kd_xy = cf.mellinger_control.kd_xy
        cf.mellinger_control.kR_xy *= 2
        cf.mellinger_control.mass = Quadrotor(State()).mass
        cf.mellinger_control.gaps_Qv *= 0.1
        cf.mellinger_control.gaps_R = 0

    cfs[1].mellinger_control.gaps_enable = True
    cfs[1].mellinger_control.gaps_eta = 1e-3

    results = [
        rollout(Quadrotor(State()), cf)
        for cf in cfs
    ]

    state_logs = [results[0][0], results[1][0], results[1][1]]
    cost_logs = [results[0][2], results[1][2]]
    names = ["default", "GAPS", "target"]

    fig, axs = plt.subplots(4, 1, figsize=(9, 9), constrained_layout=True)
    for log, name in zip(state_logs, names):
        for subplot, coord in zip(axs, [0, 2]):
            coords = [s.pos[coord] for s in log]
            subplot.plot(coords, label=name)
    for log, name in zip(cost_logs, names):
        axs[2].plot(log, label=name)
    for log, name in zip(cost_logs, names):
        axs[3].plot(np.cumsum(log), label=name)
    axs[0].set(ylabel="x")
    axs[1].set(ylabel="z")
    axs[2].set(ylabel="cost")
    axs[3].set(ylabel="cumulative cost")
    for ax in axs:
        ax.legend()
    fig.savefig("gaps_cf.pdf")

    param_logs = np.stack(results[1][3])
    T, theta_dim = param_logs.shape
    fig, axs = plt.subplots(theta_dim, 1, figsize=(2 * theta_dim, 9), constrained_layout=True)
    for trace, ax, name in zip(param_logs.T, axs, PARAM_ATTRS):
        ax.plot(trace)
        ax.set_ylabel(name)
    fig.savefig("gaps_cf_params.pdf")


if __name__ == "__main__":
    main()

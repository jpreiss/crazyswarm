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
HZ = 500


def rollout(sim: Quadrotor, cf: CrazyflieSIL):

    ticks = 0

    radius = 1.0
    period = 6.0
    omega = 2 * np.pi / period

    repeats = 4
    T = int(repeats * period * HZ) + 1

    rng = np.random.default_rng(0)
    w = 1e-2 * rng.normal(size=(T, 3))

    sim.state.vel[2] = radius * omega

    state_log = []
    target_log = []
    cost_log = []
    param_log = []
    action_log = []

    # setpoint
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    yaw = 0
    angvel = np.zeros(3)

    for t in range(T):
        tsec = ticks / (2 * HZ)
        pos[0] = radius * np.cos(omega * tsec) - radius
        pos[2] = radius * 0.5 * np.sin(2 * omega * tsec)
        vel[0] = -radius * omega * np.sin(omega * tsec)
        vel[2] = radius * 1 * omega * np.cos(2 * omega * tsec)
        acc[0] = -radius * (omega ** 2) * np.cos(omega * tsec)
        acc[2] = -radius * 2 * (omega ** 2) * np.sin(2 * omega * tsec)
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

    return state_log, target_log, cost_log, param_log, action_log


def main():
    cfs = [
        CrazyflieSIL("", np.zeros(3), "mellinger", lambda: 0)
        for _ in range(2)
    ]
    for cf in cfs:
        # without this, the simulation is unstable
        cf.mellinger_control.kd_omega_rp = 0
        cf.mellinger_control.mass = Quadrotor(State()).mass
        # because it's annoying to extract the "u"
        cf.mellinger_control.gaps_R = 0

    cfs[1].mellinger_control.gaps_enable = True
    cfs[1].mellinger_control.gaps_eta = 1e-2

    results = [
        rollout(Quadrotor(State()), cf)
        for cf in cfs
    ]

    state_logs = [results[0][0], results[1][0], results[1][1]]
    cost_logs = [results[0][2], results[1][2]]
    cost_logs = [np.array(a) for a in cost_logs]
    names = ["default", "GAPS", "target"]

    t = np.arange(len(state_logs[0])) / HZ

    W = 8
    H = 2
    def subplots(n):
        return plt.subplots(n, 1, figsize=(W, H*n), constrained_layout=True)

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
    fig_pos.savefig("gaps_cf_pos.pdf")
    fig_vel.savefig("gaps_cf_vel.pdf")

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
    fig_cost.savefig("gaps_cf_cost.pdf")

    param_logs = np.stack(results[1][3])
    T, theta_dim = param_logs.shape
    fig_param, axs_param = subplots(theta_dim)
    for trace, ax, name in zip(param_logs.T, axs_param, PARAM_ATTRS):
        ax.plot(t, trace)
        ax.set_ylabel(name)
    fig_param.savefig("gaps_cf_params.pdf")

    action_logs = [np.stack(r[4]) for r in results]
    T, ac_dim = action_logs[0].shape
    fig_action, axs_action = subplots(ac_dim)
    for log, name in zip(action_logs, names):
        for trace, ax in zip(log.T, axs_action):
            ax.plot(t, trace, label=name)
    for ax in axs_action:
        ax.legend()
    fig_action.savefig("gaps_cf_actions.pdf")


if __name__ == "__main__":
    main()

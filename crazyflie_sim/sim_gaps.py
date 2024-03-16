from copy import copy, deepcopy

import matplotlib.pyplot as plt
import numpy as np

import cffirmware
from crazyflie_sim.crazyflie_sil import CrazyflieSIL
from crazyflie_sim.backend.np import Quadrotor
from crazyflie_sim.sim_data_types import State


def rollout(sim: Quadrotor, cf: CrazyflieSIL):
    HZ = 1000

    ticks = 0
    def time_func():
        nonlocal ticks
        return ticks / float(HZ)
    # HACK!!!
    cf.time_func = time_func

    radius = 0.3
    period = 8.0
    omega = 2 * np.pi / period

    T = int(3 * period * HZ) + 1

    state_log = []
    target_log = []

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
        cf.setState(sim.state)
        cf.cmdFullState(pos, vel, acc, yaw, angvel)
        action = cf.executeController()
        f_disturb = np.zeros(3)
        sim.step(action, 1.0 / HZ, f_disturb)
        ticks += 1

    print("kp_xy:", cf.mellinger_control.gaps.kp_xy)
    print("kp_z:", cf.mellinger_control.gaps.kp_z)
    print("kd_xy:", cf.mellinger_control.gaps.kd_xy)
    print("kd_z:", cf.mellinger_control.gaps.kd_z)
    return state_log, target_log


def main():
    HZ = 1000

    cfs = [
        CrazyflieSIL("", np.zeros(3), "mellinger", lambda: 0)
        for _ in range(2)
    ]
    for cf in cfs:
        # cf.mellinger_control.ki_xy = 0
        # cf.mellinger_control.ki_z = 0
        cf.mellinger_control.kd_xy = 5
        cf.mellinger_control.gaps.kd_xy = 5
    cfs[1].mellinger_control.gaps_enable = True
    cfs[1].mellinger_control.gaps_eta = 1e-1
    results = [
        rollout(Quadrotor(State()), cf)
        for cf in cfs
    ]
    state_logs = [results[0][0], results[1][0], results[1][1]]
    names = ["default", "GAPS", "target"]

    fig, axs = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)
    for log, name in zip(state_logs, names):
        for subplot, coord in zip(axs, [0, 2]):
            coords = [s.pos[coord] for s in log]
            subplot.plot(coords, label=name)
    for ax in axs:
        ax.legend()
    fig.savefig("gaps_cf.pdf")


if __name__ == "__main__":
    main()

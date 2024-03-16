from copy import copy, deepcopy

import matplotlib.pyplot as plt
import numpy as np

import cffirmware
from crazyflie_sim.crazyflie_sil import CrazyflieSIL
from crazyflie_sim.backend.np import Quadrotor
from crazyflie_sim.sim_data_types import State


def a2v(a):
    v = cffirmware.vec3_s()
    v.x = a[0]
    v.y = a[1]
    v.z = a[2]
    return v


def a2att(a):
    att = cffirmware.attitude_t()
    att.roll = a[0]
    att.pitch = a[1]
    att.yaw = a[2]
    return att


def v2a(v):
    return np.array([v.x, v.y, v.z], dtype=np.float32)


def zero_inputs():
    setpoint = cffirmware.setpoint_t()
    setpoint.mode.x = cffirmware.modeAbs
    setpoint.mode.y = cffirmware.modeAbs
    setpoint.mode.z = cffirmware.modeAbs
    setpoint.mode.yaw = cffirmware.modeAbs
    setpoint.position = a2v(np.zeros(3))
    setpoint.yaw = 0
    setpoint.velocity = a2v(np.zeros(3))
    setpoint.attitudeRate = a2att(np.zeros(3))
    setpoint.attitude = a2att(np.zeros(3))

    state = cffirmware.state_t()
    state.position = a2v(np.zeros(3))
    state.velocity = a2v(np.zeros(3))
    state.attitude = a2att(np.zeros(3))

    sensors = cffirmware.sensorData_t()
    sensors.gyro.x = 0
    sensors.gyro.y = 0
    sensors.gyro.z = 0

    return setpoint, state, sensors


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

    setpoint, _, _ = zero_inputs()

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
    # for cf in cfs:
    #     cf.mellinger_control.ki_xy = 0
    #     cf.mellinger_control.ki_z = 0
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



def rollout_circle(ctrl):
    states = []

    dt = 1.0 / cffirmware.ATTITUDE_RATE

    setpoint, state, sensors = zero_inputs()

    for i in range(10000):
        state.position.x = 0.1


    command = cffirmware.control_t()
    command_gaps = cffirmware.control_t()


def test_gaps_circle_tracking():

    dt = 1.0 / cffirmware.ATTITUDE_RATE

    ctrl = cffirmware.controllerMellinger_t()
    ctrl_gaps = cffirmware.controllerMellinger_t()
    for c in [ctrl, ctrl_gaps]:
        cffirmware.controllerMellingerInit(c)
    ctrl_gaps.gaps_enable = 1
    ctrl_gaps.gaps_eta = 1e-3

    # introduce some error
    setpoint, state, sensors = zero_inputs()
    state.position.x = 0.1
    step = 0

    command = cffirmware.control_t()
    command_gaps = cffirmware.control_t()

    for i in range(100):
        cffirmware.controllerMellinger(
            ctrl, command, setpoint, sensors, state, step)
        cffirmware.controllerMellinger(
            ctrl_gaps, command_gaps, setpoint, sensors, state, step)
        if i == 0:
            assert control_equal(command, command_gaps)
        else:
            assert not control_equal(command, command_gaps)


if __name__ == "__main__":
    main()

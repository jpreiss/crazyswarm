from pathlib import Path
from copy import copy, deepcopy
import itertools as it

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

import rospy
import std_msgs
import tf

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


def rollout(cf, Z, timeHelper, diagonal: bool = True):
    radius = 0.75
    period = 4
    omega = 2 * np.pi / period
    init_pos = cf.initialPosition + [0, 0, Z]
    assert Z > radius / 2 + 0.2
    print("init_pos is", init_pos)

    repeats = 4
    fan_cycle = 2

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

    tf_target = tf.TransformBroadcaster()
    msg_fan = std_msgs.msg.Bool()
    msg_fan.data = False
    pub_fan = rospy.Publisher("fan", std_msgs.msg.Bool, queue_size=1)

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

        # turn the fan on or off
        if period < tsec < (repeats + 1) * period:
            repeat = int(tsec / period) - 1
            fan_on = repeat % (2 * fan_cycle) >= fan_cycle
        else:
            fan_on = False
        msg_fan.data = fan_on
        pub_fan.publish(msg_fan)

        pos[0] = radius * np.cos(omega * tsec) - radius
        pos[2] = radius * 0.5 * np.sin(2 * omega * tsec)
        if diagonal:
            pos[1] = -pos[2]
        pos += init_pos
        #print(f"{pos = }")
        #print(f"{init_pos = }")
        omega2 = tderiv * omega
        vel[0] = -radius * omega2 * np.sin(omega * tsec)
        vel[2] = radius * 1 * omega2 * np.cos(2 * omega * tsec)
        acc[0] = -radius * (omega2 ** 2) * np.cos(omega * tsec)
        acc[2] = -radius * 2 * (omega2 ** 2) * np.sin(2 * omega * tsec)
        if diagonal:
            vel[1] = -vel[2]
            acc[1] = -acc[2]

        tf_target.sendTransform(
            pos,
            [0, 0, 0, 1],  # ROS uses xyzw
            time=rospy.Time.now(),
            child="target",
            parent="world",
        )

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


def main(gaps: bool):
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

    state_log, target_log = rollout(cf, Z, timeHelper)

    cf.notifySetpointsStop()
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=2.0)
    timeHelper.sleep(3.0)

    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)

    name = "gaps.npz" if gaps else "default.npz"
    np.savez(name, state=state_log, target=target_log)


if __name__ == "__main__":
    main(gaps=False)

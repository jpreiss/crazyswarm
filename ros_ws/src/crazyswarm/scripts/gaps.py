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
from uav_trajectory import compute_omega, TrigTrajectory


def norm2(x):
    return np.sum(x ** 2)


PARAMS = [
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


def rollout(cf, Z, timeHelper, gaps, diagonal: bool = False):
    radius = 0.75
    init_pos = cf.initialPosition + [0, 0, Z]
    assert Z > radius / 2 + 0.2
    period = 4
    xtraj = TrigTrajectory.Cosine(amplitude=radius, period=period)
    ztraj = TrigTrajectory.Sine(amplitude=radius/2, period=period/2)

    repeats = 24
    fan_cycle = 6

    # setpoint
    derivs = np.zeros((4, 3))
    yaw = 0
    angvel = np.zeros(3)

    # time ramp
    t0 = timeHelper.time()
    ramp = RampTime(1.5 * period, period)
    rampdown_begin = None
    param_set = False

    tf_target = tf.TransformBroadcaster()
    msg_bool = std_msgs.msg.Bool()
    pub_fan = rospy.Publisher("fan", std_msgs.msg.Bool, queue_size=1)
    pub_trial = rospy.Publisher("trial", std_msgs.msg.Bool, queue_size=1)

    while True:
        ttrue = timeHelper.time() - t0
        tramp = ramp.val(ttrue)

        # TODO: encapsulate the ramp-down logic in the class too.
        if tramp > period and not param_set:
            cf.setParam("gaps/enable", 1 if gaps else 0)
            param_set = True

        if tramp > (repeats + 1) * period and rampdown_begin is None:
            rampdown_begin = tramp
            tderiv = 1.0
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

        trial = period < tsec < (repeats + 1) * period
        msg_bool.data = trial
        pub_trial.publish(msg_bool)

        # turn the fan on or off
        if trial:
            repeat = int(tsec / period) - 1
            # If the cycle is slow enough, run the fan for 1 extra period (e.g.
            # 3 off, 5 on) because our fans take about 1 period to start up.
            comparator = fan_cycle if fan_cycle <= 2 else fan_cycle - 1
            fan_on = repeat % (2 * fan_cycle) >= comparator
        else:
            fan_on = False
        msg_bool.data = fan_on
        pub_fan.publish(msg_bool)

        derivs[:, 0] = xtraj(tsec, timestretch=1.0/tderiv)
        derivs[:, 2] = ztraj(tsec, timestretch=1.0/tderiv)
        if diagonal:
            derivs[:, 1] = -derivs[:, 2]
        pos, vel, acc, jerk = derivs
        pos[0] -= radius
        pos += init_pos
        angvel = compute_omega(acc, jerk, yaw=0, dyaw=0)

        tf_target.sendTransform(
            pos,
            [0, 0, 0, 1],  # ROS uses xyzw
            time=rospy.Time.now(),
            child="target",
            parent="world",
        )
        cf.cmdFullState(pos, vel, acc, yaw, angvel)
        timeHelper.sleepForRate(50)


def main(bad_init: bool = False):
    gaps = rospy.get_param("crazyswarm_server/gaps")
    assert isinstance(gaps, bool)
    print("gaps is", gaps)

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Z = 0.9

    # params
    GAPS_Qv = 0.0
    GAPS_R = 0.0
    GAPS_ETA = 1e-2
    GAPS_DAMPING = 0.9995

    if bad_init:
        # detune
        full_params = ["ctrlMel/" + p for p in PARAMS]
        values = [cf.getParam(p) / 2.0 for p in full_params]
        cf.setParams(dict(zip(full_params, values)))

    if gaps:
        cf.setParams({
            "gaps/Qv": GAPS_Qv,
            "gaps/R": GAPS_R,
            "gaps/eta": GAPS_ETA,
            "gaps/damping": GAPS_DAMPING,
            "gaps/optimizer": 1,  # AdaDelta
            "gaps/ad_eps": 1e-8,
            "gaps/ad_decay": 0.95,
        })

    # Always disable in the beginning. rollout() will enable after we are up to
    # speed in the figure 8 loop.
    cf.setParam("gaps/enable", 0)

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)

    rollout(cf, Z, timeHelper, gaps=gaps)

    cf.notifySetpointsStop()
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)

    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == "__main__":
    main(bad_init=False)

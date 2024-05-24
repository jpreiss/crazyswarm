import argparse
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


PARAMS = [
    p + s for p, s in it.product(["ki_", "kp_", "kv_", "kr_", "kw_"], ["xy", "z"])
]
HZ = 500

# trajectory modes
VERT = "vert"
DIAG = "diag"
HORIZ = "horiz"


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


# polymorphism so we can run in sim
class ROSPub:
    def __init__(self):
        msgBool = std_msgs.msg.Bool
        self.msg = msgBool()
        self.pub_fan = rospy.Publisher("fan", msgBool, queue_size=1)
        self.pub_trial = rospy.Publisher("trial", msgBool, queue_size=1)
        self.tf_target = tf.TransformBroadcaster()
        self.gaps = rospy.get_param("crazyswarm_server/gaps")
        self.ada = rospy.get_param("crazyswarm_server/ada")

    def fan(self, fan: bool):
        self.msg = fan
        self.pub_fan.publish(self.msg)

    def trial(self, trial: bool):
        self.msg = trial
        self.pub_trial.publish(self.msg)

    def target(self, pos):
        self.tf_target.sendTransform(
            pos,
            [0, 0, 0, 1],  # ROS uses xyzw
            time=rospy.Time.now(),
            child="target",
            parent="world",
        )


class SimPub:
    def __init__(self):
        self.gaps = False
        self.ada = False

    def fan(self, fan):
        pass

    def trial(self, trial):
        pass

    def target(self, pos):
        pass


def rollout(cf, Z, timeHelper, pub, trajmode):
    radius = 0.75
    init_pos = cf.initialPosition + [0, 0, Z]
    assert Z > radius / 2 + 0.2
    period = 6
    traj_major = TrigTrajectory.Cosine(amplitude=radius, period=period)
    traj_minor = TrigTrajectory.Sine(amplitude=radius/2, period=period/2)

    repeats = 4
    fan_cycle = 4

    # setpoint
    derivs = np.zeros((4, 3))
    yaw = 0
    angvel = np.zeros(3)

    # time ramp
    t0 = timeHelper.time() - 1e-6  # TODO: fix div/0 error in sim
    ramp = RampTime(1.5 * period, period)
    rampdown_begin = None
    param_set = False

    while True:
        ttrue = timeHelper.time() - t0
        tramp = ramp.val(ttrue)

        # TODO: encapsulate the ramp-down logic in the class too.
        if tramp > period and not param_set:
            cf.setParam("gaps6DOF/enable", 1 if pub.gaps else 0)
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
        pub.trial(trial)

        # turn the fan on or off
        if trial:
            repeat = int(tsec / period) - 1
            # If the cycle is slow enough, run the fan for 1 extra period (e.g.
            # 3 off, 5 on) because our fans take about 1 period to start up.
            comparator = fan_cycle if fan_cycle <= 2 else fan_cycle - 1
            fan_on = repeat % (2 * fan_cycle) >= comparator
        else:
            fan_on = False
        pub.fan(fan_on)

        major = traj_major(tsec, timestretch=1.0/tderiv)
        minor = traj_minor(tsec, timestretch=1.0/tderiv)

        derivs[:, 0] = major
        if trajmode == VERT:
            derivs[:, 1] = 0
            derivs[:, 2] = minor
        elif trajmode == DIAG:
            derivs[:, 1] = minor
            derivs[:, 2] = minor
        elif trajmode == HORIZ:
            derivs[:, 1] = minor
            derivs[:, 2] = 0
        else:
            raise ValueError()

        pos, vel, acc, jerk = derivs
        pos[0] -= radius
        pos += init_pos
        angvel = compute_omega(acc, jerk, yaw=0, dyaw=0)

        pub.target(pos)

        cf.cmdFullState(pos, vel, acc, yaw, angvel)
        timeHelper.sleepForRate(50)


def main():
    # Crazyswarm's inner parser must add help to get all params.
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--detune",
        action="store_true",
        help="start with a detuned low-gain controller",
    )
    parser.add_argument(
        "--traj",
        choices=[VERT, DIAG, HORIZ],
        default=VERT,
        help="plane orientation of the figure-8",
    )
    args, _ = parser.parse_known_args()

    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Z = 0.9

    if swarm.sim:
        pub = SimPub()
    else:
        pub = ROSPub()

    print("gaps is", pub.gaps)
    print("ada is", pub.ada)

    if args.detune and not swarm.sim:
        print("Starting with detuned gains.")
        # detune
        full_params = ["gaps6DOF/" + p for p in PARAMS]
        # log space params!
        log_2 = float(np.log(2))
        values = [cf.getParam(p) - log_2 for p in full_params]
        cf.setParams(dict(zip(full_params, values)))

    # TODO: should be CLI args instead?
    if pub.gaps:
        params = {
            "eta": 2e-2,
            "optimizer": 0,  # OGD
        }
        if pub.ada:
            # AdaDelta in general will reduce the rate, so we raise it to make
            # a fair comparison.
            params["optimizer"] = 1  # adadelta
            #params["eta"] *= 2.0
            params["ad_eps"] = 1e-7
            params["ad_decay"] =  0.95
        params = {"gaps6DOF/" + k: v for k, v in params.items()}
        cf.setParams(params)

    # Always disable in the beginning. rollout() will enable after we are up to
    # speed in the figure 8 loop.
    cf.setParam("gaps6DOF/enable", 0)

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)

    rollout(cf, Z, timeHelper, pub, trajmode=args.traj)

    cf.notifySetpointsStop()
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)

    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == "__main__":
    main()

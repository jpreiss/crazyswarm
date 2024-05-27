import argparse
import itertools as it
import json
from pathlib import Path
import sys

from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
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
CIRCLE = "circle"


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
        self.prefix = rospy.get_param("crazyswarm_server/prefix")

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
        self.fan_state = False
        self.prefix = "dummy"

    def fan(self, fan):
        if fan != self.fan_state:
            print("turning fan", "on" if fan else "off")
            self.fan_state = fan

    def trial(self, trial):
        pass

    def target(self, pos):
        pass


def rollout(cf, gaps, Z, radius, timeHelper, pub, trajmode, repeats, period, fan_cycle):
    """The part of the flight where we use low-level commands."""
    radius = 0.75
    init_pos = cf.initialPosition + [0, 0, Z]
    if trajmode != HORIZ:
        assert Z > radius / 2 + 0.2
    traj_major = TrigTrajectory.Cosine(amplitude=radius, period=period)
    if trajmode == CIRCLE:
        traj_minor = TrigTrajectory.Sine(amplitude=radius, period=period)
    else:
        traj_minor = TrigTrajectory.Sine(amplitude=radius/2, period=period/2)

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
            cf.setParam("gaps6DOF/enable", 1 if gaps else 0)
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
            comparator = fan_cycle #if fan_cycle <= 2 else fan_cycle - 1
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
            derivs[:, 1] = -minor
            derivs[:, 2] = minor
        elif trajmode in [HORIZ, CIRCLE]:
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
    group = parser.add_argument_group("GAPS experiment params", "")
    group.add_argument(
        "--gaps",
        action="store_true",
        help="enable GAPS.",
    )
    group.add_argument(
        "--detune",
        action="store_true",
        help="start with a detuned low-gain controller.",
    )
    group.add_argument(
        "--traj",
        choices=[VERT, DIAG, HORIZ],
        default=VERT,
        help="plane orientation of the figure-8.",
    )
    group.add_argument(
        "--repeats",
        type=int,
        default=4,
        help="number of figure-8 cycles to fly.",
    )
    group.add_argument(
        "--period",
        type=int,
        default=6,
        help="duration of one figure-8 cycle.",
    )
    group.add_argument(
        "--fan_cycle",
        type=int,
        default=-1,
        help=(
            "number of cycles in high/low phase of fan "
            "(so actually a half cycle). -1 means no fan."
        ),
    )
    args, _ = parser.parse_known_args()

    swarm = Crazyswarm(parent_parser=parser)
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Z = 0.9
    radius = 0.75

    if swarm.sim:
        pub = SimPub()
    else:
        pub = ROSPub()

    print("file prefix:", pub.prefix)
    print("config:")
    for k, v in vars(args).items():
        print(f"    {k}: {v}")

    path = Path.home() / ".ros" / (pub.prefix + "_config.json")
    with open(path, "w") as f:
        json.dump(vars(args), f)

    if args.detune and not swarm.sim:
        print("Starting with detuned gains.")
        # detune
        full_params = ["gaps6DOF/" + p for p in PARAMS]
        # log space params!
        log_2 = float(np.log(2))
        values = [cf.getParam(p) - log_2 for p in full_params]
        cf.setParams(dict(zip(full_params, values)))

    if args.gaps:
        params = {
            "eta": 2e-2,
            "optimizer": 0,  # OGD
        }
        params = {"gaps6DOF/" + k: v for k, v in params.items()}
        cf.setParams(params)

    fan_cycle = args.fan_cycle
    if fan_cycle == -1:
        fan_cycle = 2 * args.repeats * args.period + 1

    # Always disable in the beginning. rollout() will enable after we are up to
    # speed in the figure 8 loop.
    cf.setParam("gaps6DOF/enable", 0)

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)

    rollout(cf, args.gaps, Z, radius, timeHelper, pub, trajmode=args.traj,
        repeats=args.repeats, period=args.period, fan_cycle=fan_cycle)

    cf.notifySetpointsStop()
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)

    cf.land(targetHeight=0.05, duration=Z+1.0)
    timeHelper.sleep(Z+2.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python

"""Example of using uav_trajectory.TrigTrajectory.

Flies a circle with constant body-frame roll/pitch and rotating yaw, like a
fixed-wing, or a human piloting a rotorcraft from first-person view.
"""

import itertools as it

import numpy as np

from pycrazyswarm import *
from uav_trajectory import TrigTrajectory


def circle(timeHelper, cf, z, radius=0.1, period=6, loops=1, rate=50):
    init = np.array(cf.initialPosition)
    center = init + [-radius, 0, z]
    xtraj = TrigTrajectory.Cosine(amplitude=radius, period=period)
    ytraj = TrigTrajectory.Sine(amplitude=radius, period=period)
    yawscale = 2 * np.pi / period

    t0 = timeHelper.time()
    while True:
        t = timeHelper.time() - t0
        if t > period * loops:
            return
        xderivs = xtraj(t)
        yderivs = ytraj(t)
        pos = np.array([xderivs[0], yderivs[0], 0])
        vel = np.array([xderivs[1], yderivs[1], 0])
        acc = np.array([xderivs[2], yderivs[2], 0])
        yaw = yawscale * t
        omega = np.array([0, 0, yawscale])
        cf.cmdFullState(pos + center, vel, acc, yaw, omega)
        timeHelper.sleepForRate(rate)


PARAMS = [
    p + s for p, s in it.product(["ki_", "kp_", "kv_", "kr_", "kw_"], ["xy", "z"])
]

def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    # log space params!
    full_params = ["gaps6DOF/" + p for p in PARAMS]
    values = [cf.getParam(p) - float(np.log(2)) for p in full_params]
    cf.setParams(dict(zip(full_params, values)))

    if True:
        params = {
            "eta": 2e-2,
            "optimizer": 0,  # OGD
            "enable": 1,
        }
        params = {"gaps6DOF/" + k: v for k, v in params.items()}
        cf.setParams(params)



    rate = 30.0
    Z = 0.99
    radius = 0.65

    periods = [5, 4.5, 4, 3.5,  3, 2.75, 2.5, 2.25, 2]
    periods = periods + periods[::-1]

    cf.takeoff(targetHeight=Z, duration=Z+1.0)
    timeHelper.sleep(Z+1.0)
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=2.0)
    timeHelper.sleep(3.0)

    for period in periods:
        circle(timeHelper, cf, Z, radius=radius, period=period)

    cf.notifySetpointsStop()
    timeHelper.sleep(0.01)
    cf.goTo(cf.initialPosition + [0, 0, Z], yaw=0, duration=1.0)
    timeHelper.sleep(2.0)
    cf.land(targetHeight=0.03, duration=Z+1.0)
    timeHelper.sleep(Z+1.0)


if __name__ == "__main__":
    main()

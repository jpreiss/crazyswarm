#!/usr/bin/env python

import numpy as np
from pycrazyswarm import *


Z = 0.3
sleepRate = 30


def goCircle(timeHelper, cf, totalTime, radius, kPosition, iterations=1):
        startPos = cf.position()
        centerCircle = startPos - np.array([radius, 0, 0])

        startTime = timeHelper.time()
        time = 0.0
        duration = totalTime * iterations

        while time < duration:
            time = timeHelper.time() - startTime
            omega = 2 * np.pi / totalTime
            pos = centerCircle + radius * np.array(
                [np.cos(omega * time), np.sin(omega * time), 0])
            vx = -radius * omega * np.sin(omega * time)  
            vy = radius * omega * np.cos(omega * time)
            vel = [vx, vy, 0.0]
            cf.cmdFullState(pos, vel, acc=np.zeros(3), yaw=0.0, omega=np.zeros(3))
            timeHelper.sleepForRate(sleepRate)


if __name__ == "__main__":
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    cf.takeoff(targetHeight=Z, duration=1.0+Z)
    timeHelper.sleep(2.0 + Z)

    cf.goTo([-8.0, 0.0, 0.0], yaw=0.0, duration=12.0, relative=True)
    timeHelper.sleep(2.0)
    # Preempt!
    goCircle(timeHelper, cf, totalTime=8, radius=1, kPosition=1)

    # If preemption worked correctly, we should stay at the start/end point of the circle. If it is broken, we should jump to the goTo destination.
    timeHelper.sleep(2.5)

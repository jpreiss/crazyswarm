"""Takeoff-hover-land for one CF. Useful to validate hardware config."""

import numpy as np

from pycrazyswarm import Crazyswarm


TAKEOFF_DURATION = 2.5
HOVER_DURATION = 5.0


def main():
    swarm = Crazyswarm()
    timeHelper = swarm.timeHelper
    cf = swarm.allcfs.crazyflies[0]

    Z = 1.0

    cf.takeoff(targetHeight=Z, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1)

    pos = np.array(cf.initialPosition) + np.array([0, 0, Z])
    cf.goTo(pos, 0, duration=1.0)
    timeHelper.sleep(1)

    # yaw
    halfpi = np.pi / 2
    duration = 0.5
    for i in range(1, 5):
        cf.goTo(pos, i * halfpi, duration)
        timeHelper.sleep(duration)

    cf.land(targetHeight=0.04, duration=TAKEOFF_DURATION)
    timeHelper.sleep(TAKEOFF_DURATION + 1)


if __name__ == "__main__":
    main()

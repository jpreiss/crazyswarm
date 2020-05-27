"""Demonstrate keyboard input."""

import numpy as np

from pycrazyswarm import *


crazyflies_yaml = """
crazyflies:
- id: 1
  channel: 110
  initialPosition: [-1.0, 0.0, 0.0]
- id: 2
  channel: 110
  initialPosition: [1.0, 1.0, 0.0]
- id: 3
  channel: 110
  initialPosition: [0.0, 1.0, 0.0]
- id: 4
  channel: 110
  initialPosition: [1.0, 0.0, 0.0]
- id: 5
  channel: 110
  initialPosition: [0.0, 0.0, 0.0]
"""

RADII = np.array([0.125, 0.125, 0.375])
# Since the user-controlled agent won't respect its Voronoi cell,
# we use a larger collision volume so others get out of the way sooner.
SAFETY_FACTOR = 0.125

if __name__ == "__main__":

    swarm = Crazyswarm(crazyflies_yaml=crazyflies_yaml)
    timeHelper = swarm.timeHelper
    cfs = swarm.allcfs.crazyflies

    # The user will control one Crazyflie with the keyboard.
    user = cfs[0]
    user.setLEDColor(0.0, 1.0, 1.0)

    robots = cfs[1:]
    for robot in robots:
        others = [cf for cf in cfs if cf.id != robot.id]
        robot.enableCollisionAvoidance(others, RADII + SAFETY_FACTOR)
        robot.setLEDColor(0.0, 0.0, 1.0)
        robot.cmdPosition(robot.initialPosition, yaw=0.0)

    visualizer = timeHelper.visualizer
    visualizer.showEllipsoids(RADII)

    # WASD keys match the point of view from the Crazyswarm warehouse desk.
    keyDirs = {
        "w": np.array([ 1.0,  0.0, 0.0]),
        "a": np.array([ 0.0,  1.0, 0.0]),
        "s": np.array([-1.0,  0.0, 0.0]),
        "d": np.array([ 0.0, -1.0, 0.0]),
    }

    # To avoid instantaneous starts and stops, keys are treated as acceleration
    # inputs. Heavy damping makes keys feel more like smoothed velocity inputs.
    velocity = np.zeros(3)

    while True:

        # Handle simultaneous keys correctly.
        acc = np.zeros(3)
        for key, direction in keyDirs.items():
            if visualizer.keyState[key]:
                acc += direction
        if np.any(acc != 0.0):
            acc *= 2.0 * timeHelper.dt / np.linalg.norm(acc)

        velocity = 0.7 * velocity + acc
        user.cmdVelocityWorld(velocity, 0.0)
        timeHelper.sleep(timeHelper.dt)

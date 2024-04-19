import numpy as np

from uav_trajectory import TrigTrajectory


# Error due to using complex exponential instead of trigonometric functions
# internally should be minimal.
def _veryclose(a, b):
    return np.isclose(a, b, rtol=1e-10, atol=1e-16)


def test_TrigTrajectory_deriv():
    rng: np.random.Generator = np.random.default_rng(0)
    epsilon = 1e-6
    for _ in range(1000):
        # make sure we get past 2pi often
        amp, omega, phase, t = 2 * np.pi * rng.exponential(size=4)
        traj = TrigTrajectory(amplitude=amp, omega=omega, phase=phase, deriv=5)
        ft = traj(t)
        assert _veryclose(ft[0], amp * np.cos(omega * t + phase))
        fback = traj(t - epsilon)
        ffwd = traj(t + epsilon)
        appxderivs = (ffwd - fback) / (2 * epsilon)
        assert np.allclose(ft[1:], appxderivs[:-1])


def test_TrigTrajectory_phase():
    tsin = TrigTrajectory.Sine(omega=1)
    for t in np.linspace(-100, 100, 1000):
        assert _veryclose(tsin(t)[0], np.sin(t))


# TODO: test timestretch function

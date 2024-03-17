from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from ..sim_data_types import Action, State
import fastrowan as rowan


class Quadrotor:
    """Basic rigid body quadrotor model (no drag) using numpy and rowan."""

    def __init__(self, state):
        # parameters (Crazyflie 2.0 quadrotor)
        self.mass = 0.034  # kg
        # self.J = np.array([
        # 	[16.56,0.83,0.71],
        # 	[0.83,16.66,1.8],
        # 	[0.72,1.8,29.26]
        # 	]) * 1e-6  # kg m^2
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])

        # Note: we assume here that our control is forces
        arm_length = 0.046  # m
        arm = 0.707106781 * arm_length
        t2t = 0.006  # thrust-to-torque ratio
        self.B0 = np.array([
            [1, 1, 1, 1],
            [-arm, -arm, arm, arm],
            [-arm, arm, arm, -arm],
            [-t2t, t2t, -t2t, t2t]
            ])
        self.g = 9.81  # not signed

        if self.J.shape == (3, 3):
            self.inv_J = np.linalg.pinv(self.J)  # full matrix -> pseudo inverse
        else:
            self.inv_J = 1 / self.J  # diagonal matrix -> division

        # solve for the quadratic drag coefficient based on empirical data
        THRUST_WEIGHT = 2.0
        TOP_SPEED = 10.0
        forward_thrust = self.mass * self.g * np.sqrt(THRUST_WEIGHT - 1)
        self.drag_linear = forward_thrust / (TOP_SPEED ** 2)

        self.state = state

    def step(self, action, dt, f_a=np.zeros(3)):

        # convert RPM -> Force
        def rpm_to_force(rpm):
            # polyfit using data and scripts from https://github.com/IMRCLab/crazyflie-system-id
            p = [2.55077341e-08, -4.92422570e-05, -1.51910248e-01]
            force_in_grams = np.polyval(p, rpm)
            force_in_newton = force_in_grams * 9.81 / 1000.0
            return np.maximum(force_in_newton, 0)

        force = rpm_to_force(action.rpm)

        vel = self.state.vel
        linear_damping = -self.drag_linear * norm(self.state.vel) * self.state.vel
        angular_damping = -0.01 * norm(self.state.omega) * self.state.omega

        # compute next state
        eta = np.dot(self.B0, force)
        f_u = np.array([0, 0, eta[0]])
        tau_u = np.array([eta[1], eta[2], eta[3]])

        # dynamics
        # dot{p} = v
        pos_next = self.state.pos + self.state.vel * dt
        # mv = mg + R f_u + f_a
        vel_next = self.state.vel + (
            np.array([0, 0, -self.g]) +
            (linear_damping + rowan.rotate(self.state.quat, f_u) + f_a) / self.mass) * dt

        # dot{R} = R S(w)
        # to integrate the dynamics, see
        # https://www.ashwinnarayan.com/post/how-to-integrate-quaternions/, and
        # https://arxiv.org/pdf/1604.08139.pdf
        # Sec 4.5, https://arxiv.org/pdf/1711.02508.pdf
        omega_global = rowan.rotate(self.state.quat, self.state.omega)
        q_next = rowan.normalize(
            rowan.calculus.integrate(
                self.state.quat, omega_global, dt))

        # mJ = Jw x w + tau_u
        omega_next = self.state.omega + (
            self.inv_J * (angular_damping + np.cross(self.J * self.state.omega, self.state.omega) + tau_u)) * dt

        self.state.pos = pos_next
        self.state.vel = vel_next
        self.state.quat = q_next
        self.state.omega = omega_next

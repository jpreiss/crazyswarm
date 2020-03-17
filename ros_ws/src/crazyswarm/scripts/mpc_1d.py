from __future__ import print_function
import time

import cvxpy as cvx
import matplotlib.pyplot as plt
import numpy as np

print("Using cvxpy v", cvx.__version__)

dim = 1
T = 10
dt = 0.1

p = cvx.Variable((T+1, dim))
# Last velocity and accel values are not used, but kept for constraint building simplicity.
v = cvx.Variable((T+1, dim))
a = cvx.Variable((T+1, dim))

c_a = 0.1
c_v = 1.0
c_p = 10.0
c_vT = 100.0
c_pT = 1000.0
a_max = 3.0
slew_max = 10.0 * dt

goal = cvx.Parameter(dim)
p0 = cvx.Parameter(dim)
v0 = cvx.Parameter(dim)
a0 = cvx.Parameter(dim)

cost = c_pT * cvx.sum_squares(p[-1,:] - goal) + c_vT * cvx.sum_squares(v[-1,:])
constr = [
    p[0,:] == p0,
    v[0,:] == v0,
    #v[-2,:] == [0.0] * dim,
    a[0,:] == a0,
    #a[-2,:] == [0.0] * dim,
]
for t in range(T):
    cost += (
        c_p * cvx.sum_squares(p[t,:] - goal) + 
        c_v * cvx.sum_squares(v[t,:]) + 
        c_a * cvx.sum_squares(a[t+1,:])
    )
    constr += [
        p[t+1,:] == p[t,:] + dt * v[t,:] + dt**2 * a[t+1,:],
        v[t+1,:] == v[t,:] + dt * a[t+1,:],
        cvx.norm(a[t+1,:], "inf") <= a_max,
        cvx.norm(a[t+1,:] - a[t,:], "inf") <= slew_max,
    ]

problem = cvx.Problem(cvx.Minimize(cost), constr)

for p1 in (0.1, 0.5, 1.0, 2.0):
    goal.value = [p1]
    p0.value = [0.0]
    v0.value = [0.0]
    a0.value = [0.0]
    problem.solve(verbose=True)

    fig, axs = plt.subplots(3, 1)

    for ax in axs:
        ax.set_xlim(0.0, T)

    a_vals = list(a.value[1:]) + [a.value[-1]]
    axs[0].step(np.arange(T+1), a_vals, where="post")
    axs[0].set_ylabel("acceleration")
    axs[0].axhline(0.0, linestyle="--")
    axs[0].axhline(-a_max, linestyle="-.")
    axs[0].axhline(a_max, linestyle="-.")

    axs[1].plot(v.value)
    axs[1].set_ylabel("velocity")
    axs[1].axhline(0.0, linestyle="--")

    axs[2].plot(p.value)
    axs[2].set_ylabel("position")
    axs[2].axhline(goal.value, linestyle="--")
    plt.show()

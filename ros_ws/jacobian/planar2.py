from collections import namedtuple

import colorama
import numpy as np


def namedvec(name, fields, sizes):
    fields = fields.split(" ")
    sizes = sizes.split(" ")
    assert len(fields) == len(sizes)
    sizes = [int(s) for s in sizes]
    base = namedtuple(name, fields)

    def to_arr(self):
        for v in [*self]:
            assert isinstance(v, np.ndarray) or isinstance(v, float)
        return np.block([*self])

    splits = np.cumsum(sizes)
    @classmethod
    def from_arr(cls, arr):
        blocks = map(np.squeeze, np.split(arr, splits[:-1]))
        return cls(*blocks)

    @classmethod
    def dim_str(cls, dim):
        idx = np.argmin(dim >= splits)
        inner_idx = dim - splits[idx] + sizes[idx]
        return f"{fields[idx]}[{inner_idx}]"

    return type(
        name,
        (base,),
        dict(to_arr=to_arr, from_arr=from_arr, dim_str=dim_str),
    )


State = namedvec("State", "ierr p v r w", "2 2 2 1 1")
Action = namedvec("Action", "thrust torque", "1 1")
Target = namedvec("Target", "p_d v_d a_d w_d", "2 2 2 1")
Param = namedvec("Param", "ki kp kv kr kw", "1 1 1 1 1")
Const = namedvec("Const", "g m j dt", "1 1 3 1")


def outer(a, b):
    return a[:, None] @ b[None, :]


def angleto(a, b):
    """Returns angle rotating vec. a to vec. b and gradients.

    Vectors must be unit length.

    Returns: angle, dangle/da, dangle/db
    """
    # compute atan2 of b in the coordinate frame of a.
    ax, ay = a
    bx, by = b
    R = np.array([
        [ ax, ay],
        [-ay, ax],
    ])
    Rb = R @ b 
    DRb_a = np.array([
        [bx,  by],
        [by, -bx]
    ])
    angle = np.arctan2(Rb[1], Rb[0])
    grad_atan2 = np.array([-Rb[1], Rb[0]])
    grad_a = grad_atan2.T @ DRb_a
    grad_b = grad_atan2.T @ R
    assert grad_a.shape[-1] == 2
    assert grad_b.shape[-1] == 2

    return angle, grad_a, grad_b


def ctrl(x: State, xd: Target, th: Param, c: Const):
    """Returns: u, next x, Du_x, Du_th, Dx_x, Dx_u."""
    # derived state
    up = np.array([-np.sin(x.r), np.cos(x.r)])
    Dup_x = np.array([
        [0, 0, 0, 0, 0, 0, -np.cos(x.r), 0],
        [0, 0, 0, 0, 0, 0, -np.sin(x.r), 0],
    ])
    assert Dup_x.shape[1] == len(x.to_arr())
    g = np.array([0, c.g])

    # double-check the derivatives
    def up_fn(x):
        r = x[-2]
        return np.array([-np.sin(r), np.cos(r)])
    xa = x.to_arr()
    finitediff_check(xa, Dup_x, up_fn, lambda i: "TODO")


    # CONTROLLER

    # ----------
    # position part components
    feedback = (
        - th.ki * x.ierr
        - th.kp * (x.p - xd.p_d)
        - th.kv * (x.v - xd.v_d)
    )
    a = feedback + xd.a_d + g
    Da_x = np.array([
        [-th.ki,      0, -th.kp,      0, -th.kv,      0, 0, 0],
        [     0, -th.ki,      0, -th.kp,      0, -th.kv, 0, 0],
    ])
    Da_th = np.array([
        [-x.ierr[0], -x.p[0], -x.v[0], 0, 0],
        [-x.ierr[1], -x.p[1], -x.v[1], 0, 0],
    ])

    thrust = np.linalg.norm(a)
    Dthrust_a = (a / thrust).reshape((1, 2))

    upgoal = a / thrust
    Dupgoal_a = (1.0 / thrust) * np.eye(2) - (1 / thrust ** 3) * outer(a, a)

    # attitude part components
    er, Der_up, Der_upgoal = angleto(up, upgoal)

    # double-check the derivatives
    def angleto_lambda(xflat):
        a, b = xflat.reshape((2, 2))
        return angleto(a, b)[0]
    D = np.concatenate([Der_up, Der_upgoal])[None, :]
    finitediff_check(np.concatenate([up, upgoal]), D, angleto_lambda, lambda i: "TODO")

    ew = x.w - xd.w_d
    torque = -th.kr * er - th.kw * ew
    u = Action(thrust=thrust, torque=torque)

    # controller chain rules
    Dthrust_x = Dthrust_a @ Da_x
    Dthrust_th = Dthrust_a @ Da_th

    Der_x = Der_up @ Dup_x + Der_upgoal @ Dupgoal_a @ Da_x
    Der_th = Der_upgoal @ Dupgoal_a @ Da_th
    Dtorque_xw = np.array([
        [0, 0, 0, 0, 0, 0, 0, -th.kw],
    ])
    Dtorque_x = -th.kr * Der_x + Dtorque_xw
    Dtorque_th = np.array([[0, 0, 0, -er, -ew]])
    Du_x = np.block([
        [Dthrust_x],
        [Dtorque_x],
    ])
    Du_th = np.block([
        [Dthrust_th],
        [Dtorque_th],
    ])
    return u, Du_x, Du_th


def dynamics(x: State, xd: Target, u: Action, c: Const):
    # DYNAMICS
    # --------

    # derived state - TODO factor out?
    up = np.array([-np.sin(x.r), np.cos(x.r)])
    Dup_x = np.array([
        [0, 0, 0, 0, 0, 0, -np.cos(x.r), 0],
        [0, 0, 0, 0, 0, 0, -np.sin(x.r), 0],
    ])
    g = np.array([0, c.g])

    # Normally I would use symplectic Euler integration, but plain forward
    # Euler gives simpler Jacobians.

    acc = u.thrust * up - g
    Dacc_x = u.thrust * Dup_x
    x_t = State(
        ierr = x.ierr + c.dt * (x.p - xd.p_d),
        p = x.p + c.dt * x.v,
        v = x.v + c.dt * acc,
        r = x.r + c.dt * x.w,
        w = x.w + c.dt * u.torque,
    )
    Dvt_r = c.dt * Dacc_x[:, [-2]]
    I2 = np.eye(2)
    I1 = np.eye(1)
    Z22 = np.zeros((2, 2))
    Z21 = np.zeros((2, 1))
    Z12 = np.zeros((1, 2))
    Z11 = np.zeros((1, 1))
    dt2 = c.dt * I2
    dt1 = c.dt * I1
    Dx_x = np.block([
        [ I2, dt2, Z22,   Z21, Z21],
        [Z22,  I2, dt2,   Z21, Z21],
        [Z22, Z22,  I2, Dvt_r, Z21],
        [Z12, Z12, Z12,    I1, dt1],
        [Z12, Z12, Z12,   Z11,  I1],
    ])
    Dx_u = np.block([
        [           0,    0],
        [           0,    0],
        [           0,    0],
        [           0,    0],
        [c.dt * up[0],    0],
        [c.dt * up[1],    0],
        [           0,    0],
        [           0, c.dt],
    ])

    return x_t, Dx_x, Dx_u


# TODO: costs?


def random_state(rng):
    return State(
        ierr = 0.1 * rng.normal(size=2),
        p = 1.0 * rng.normal(size=2),
        v = 0.1 * rng.normal(size=2),
        r = 0.1 * rng.uniform(-np.pi / 3, np.pi / 3),
        w = 0.1 * rng.normal(),
    )


def random_target(rng):
    return Target(
        p_d = 0.1 * rng.normal(size=2),
        v_d = 0.1 * rng.normal(size=2),
        a_d = 0.1 * rng.normal(size=2),
        w_d = 0.1 * rng.normal(),
    )


def random_param(rng):
    return Param.from_arr(rng.uniform(0.1, 4, size=5))


EPS = 1e-8
RTOL = 1e-4
ATOL = 1e-6

def color_rtol(x):
    if abs(x) > RTOL:
        return f"{colorama.Fore.RED}{x:.4e}{colorama.Fore.RESET}"
    return f"{x:.4e}"

def color_atol(x):
    if abs(x) > ATOL:
        return f"{colorama.Fore.RED}{x:.4e}{colorama.Fore.RESET}"
    return f"{x:.4e}"


def finitediff_check(x, D, f, dim_str):
    n = x.size
    assert D.shape[1] == n
    y = f(x)

    for i in range(n):
        dx = 0 * x
        dx[i] += EPS
        y2 = f(x + dx)
        if False:
            y2_pred = y + D @ dx
            error = y2_pred - y2
            print(f"{y = }\n{y2 = }\n{y2_pred = }\n{error = }")
            assert np.allclose(y2_pred, y2, rtol=1e-3, atol=1e-3)
        D_finite = (y2 - y) / EPS
        D_analytic = D[:, i]
        aerr = D_finite - D_analytic
        rerr = aerr / (D_analytic + (D_analytic == 0))
        ok = np.allclose(D_finite, D_analytic, rtol=RTOL, atol=ATOL)
        if not ok:
            print(f"wrt input {dim_str(i)}")
            print(f"{D_finite = }\n{D_analytic = }")
            with np.printoptions(formatter=dict(float=color_atol)):
                print(f"{aerr = }")
            with np.printoptions(formatter=dict(float=color_rtol)):
                print(f"{rerr = }")
            assert False


def main():
    const = Const(g=9.81, m=1, j=None, dt=0.01)
    rng = np.random.default_rng(0)
    for i in range(100):
        x = random_state(rng)
        xd = random_target(rng)
        th = random_param(rng)

        #x2 = random_state(rng)
        #xd2 = random_target(rng)
        #th2 = random_param(rng)

        #dx = np.array([*x2]) - np.array([*x])
        #dxd = np.array([*xd2]) - np.array([*xd])
        #dth = np.array([*th2]) - np.array([*th])

        xa = x.to_arr()

        u, Du_x, Du_th = ctrl(x, xd, th, const)
        xt, Dx_x, Dx_u = dynamics(x, xd, u, const)

        print(f"{x = }\n{xd = }\n{th = }\n{u = }")

        def ctrl_x2u(xa):
            x2 = State.from_arr(xa)
            return ctrl(x2, xd, th, const)[0].to_arr()

        def ctrl_th2u(tha):
            th2 = Param.from_arr(tha)
            return ctrl(x, xd, th2, const)[0].to_arr()

        print("du/dx")
        finitediff_check(x.to_arr(), Du_x, ctrl_x2u, State.dim_str)
        #print("du/dth")
        #finitediff_check(th.to_arr(), Du_th, ctrl_th2u)

        def dyn_x2x(xa):
            x2 = State.from_arr(xa)
            return dynamics(x2, xd, u, const)[0].to_arr()

        def dyn_u2x(ua):
            u2 = Action.from_arr(ua)
            return dynamics(x, xd, u2, const)[0].to_arr()

        print("dx/dx")
        finitediff_check(x.to_arr(), Dx_x, dyn_x2x, State.dim_str)

        print("dx/du")
        finitediff_check(u.to_arr(), Dx_u, dyn_u2x, Action.dim_str)


if __name__ == "__main__":
    main()

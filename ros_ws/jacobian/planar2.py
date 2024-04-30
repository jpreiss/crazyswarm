from collections import namedtuple

import colorama
import numpy as np

from planar import angleto
from SO3 import error


def up_to_R3(up):
    """Converts an 'up' vector to a 3d rotation matrix, plus jacobian."""
    R = np.array([
        [ up[1], 0, up[0]],
        [    0,  1,     0],
        [-up[0], 0, up[1]],
    ])
    J = np.array([
        [ 0, 1],
        [ 0, 0],
        [-1, 0],
        [ 0, 0],
        [ 0, 0],
        [ 0, 0],
        [ 1, 0],
        [ 0, 0],
        [ 0, 1],
    ])
    JR = (J @ up).reshape((3, 3)).T
    JR[1, 1] = 1
    assert np.all(R.flat == JR.flat)
    return R, J


def namedvec(name, fields, sizes):
    """Namedtuple plus helpers for going to/from concatenated arrays."""
    fields = fields.split(" ")
    sizes = sizes.split(" ")
    assert len(fields) == len(sizes)
    sizes = [int(s) for s in sizes]
    splits = np.cumsum(sizes)
    base = namedtuple(name, fields)

    def to_arr(self):
        for v in [*self]:
            assert isinstance(v, np.ndarray) or isinstance(v, float)
        return np.block([*self])

    @classmethod
    def from_arr(cls, arr):
        blocks = map(np.squeeze, np.split(arr, splits[:-1]))
        return cls(*blocks)

    @classmethod
    def dim_str(cls, dim):
        """Converts index into the concatenated vector to subfield index."""
        idx = np.argmin(dim >= splits)
        inner_idx = dim - splits[idx] + sizes[idx]
        return f"{fields[idx]}[{inner_idx}]"

    return type(
        name,
        (base,),
        dict(size=np.sum(sizes), to_arr=to_arr, from_arr=from_arr, dim_str=dim_str),
    )


State = namedvec("State", "ierr p v R w", "3 3 3 9 3")
Action = namedvec("Action", "thrust torque", "1 3")
Target = namedvec("Target", "p_d v_d a_d w_d", "3 3 3 3")
Param = namedvec("Param", "ki kp kv kr kw", "1 1 1 1 1")
Const = namedvec("Const", "g m j dt", "1 1 3 1")


def ctrl(x: State, xd: Target, th: Param, c: Const):
    """Returns: u, Du_x, Du_th."""
    g = np.array([0, 0, c.g])

    # CONTROLLER

    # ----------
    # position part components
    perr = x.p - xd.p_d
    verr = x.v - xd.v_d
    feedback = - th.ki * x.ierr - th.kp * perr - th.kv * verr
    a = feedback + xd.a_d + g
    I = np.eye(3)
    Da_x = np.block([
        [-th.ki * I, -th.kp * I, -th.kv * I, 0 * I]
    ])
    Da_th = np.block([
        [-x.ierr[:, None], -perr[:, None], -verr[:, None], 0 * I]
    ])

    # double-check the derivatives
    def a_fn(th2):
        th2 = Param.from_arr(th2)
        feedback = (
            - th2.ki * x.ierr
            - th2.kp * (x.p - xd.p_d)
            - th2.kv * (x.v - xd.v_d)
        )
        return feedback + xd.a_d + g
    finitediff_check(th.to_arr(), Da_th, a_fn, Param.dim_str)

    thrust = np.linalg.norm(a)
    Dthrust_a = (a / thrust).reshape((1, 2))

    zgoal = a / thrust
    Dzgoal_a = (1.0 / thrust) * np.eye(3) - (1 / thrust ** 3) * np.outer(a, a)
    ygoal = np.array([0, 1, 0])
    Dygoal_a = np.zeros(3)
    Dxgoal_zgoal = np.array([
        [ 0, 0, 1],
        [ 0, 0, 0],
        [-1, 0, 0],
    ])
    xgoal = Dxgoal_zgoal @ zgoal
    Dxgoal_a = Dxgoal_zgoal @ Dzgoal_a
    Rd3 = np.stack([xgoal, ygoal, zgoal]).T
    DRd3_a = np.block([
        [Dxgoal_a],
        [Dygoal_a],
        [Dzgoal_a],
    ], axis=0)

    # attitude part components
    R3 = x.R.reshape((3, 3)).T
    Z93 = np.zeros((9, 3))
    DR3_x = np.block([Z93, Z93, Z93, np.eye(9), Z93])
    er3, Der3_R3, Der3_Rd3 = error(R3, Rd3)
    assert np.isclose(er3[0], 0)
    assert np.isclose(er3[2], 0)

    erold, *_ = angleto(upgoal, up)
    assert np.sign(erold) == np.sign(er3[1])

    # double-check the derivatives
    # def angleto_lambda(xflat):
    #     a, b = xflat.reshape((2, 2))
    #     return angleto(a, b)[0]
    # D = np.concatenate([Der_upgoal, Der_up])[None, :]
    # finitediff_check(np.concatenate([upgoal, up]), D, angleto_lambda, lambda i: "vecs")

    ew = x.w - xd.w_d
    torque = -th.kr * er3 - th.kw * ew
    u = Action(thrust=thrust, torque=torque)

    # controller chain rules
    Dthrust_x = Dthrust_a @ Da_x
    Dthrust_th = Dthrust_a @ Da_th

    #Der_x = Der_up @ Dup_x + Der_upgoal @ Dupgoal_a @ Da_x
    Der3_x = Der3_R3 @ DR3_x + Der3_Rd3 @ DRd3_a @ Da_x

    #Der_th = Der_upgoal @ Dupgoal_a @ Da_th
    Der3_th = Der3_Rd3 @ DRd3_a @ Da_th

    Dtorque_xw = np.zeros((3, State.size))
    Dtorque_xw[:, -3:] = -th.kw * np.eye(3)
    Dtorque_x = -th.kr * Der3_x + Dtorque_xw

    #Dtorque_th = -th.kr * Der_th + np.array([[0, 0, 0, -er, -ew]])
    Z31 = np.zeros((3, 1))
    Dtorque_th = -th.kr * Der3_th + np.block([[Z31, Z31, Z31, -er3, -ew]])

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

    R = x.R.reshape((3, 3)).T
    up = R[:, 2]
    Z33 = np.zeros((3, 3))
    Dup_x = np.block([[Z33, Z33, Z33, Z33, Z33, np.eye(3), Z33]])
    g = np.array([0, 0, c.g])

    # Normally I would use symplectic Euler integration, but plain forward
    # Euler gives simpler Jacobians.

    acc = u.thrust * up - g
    Dacc_x = u.thrust * Dup_x
    x_t = State(
        ierr = x.ierr + c.dt * (x.p - xd.p_d),
        p = x.p + c.dt * x.v,
        v = x.v + c.dt * acc,
        #r = x.r + c.dt * x.w,
        # TODO: R
        w = x.w + c.dt * u.torque,
    )
    # TODO: This became trivial after we went from angle state to rotation
    # matrix -- condense some ops.
    Dvt_R = c.dt * Dacc_x[:, 9:-3]
    assert Dvt_R.shape == (3, 9)

    I3 = np.eye(3)
    I9 = np.eye(9)
    Z31 = np.zeros((3, 1))
    Z33 = np.zeros((3, 3))
    Z39 = np.zeros((3, 9))
    Z93 = Z39.T

    Rx, Ry, Rz = R.T
    DR_w = c.dt * np.block([
        [Z31, -Rz,  Ry],
        [ Rz, Z31, -Rx],
        [-Ry,  Rx, Z32],
    ])
    assert DR_w.shape == (9, 3)

    dt3 = c.dt * I3
    Dx_x = np.block([
        [ I3, dt3, Z33,   Z39,  Z21],
        [Z33,  I3, dt3,   Z39,  Z21],
        [Z33, Z33,  I3, Dvt_R,  Z21],
        [Z93, Z93, Z93,    I9, DR_w],
        [Z33, Z33, Z33,   Z39,   I3],
    ])
    # (Refers to Dx_x construction above.) Skipping Coriolis term that would
    # make dw'/dw nonzero because it requires system ID of the inertia matrix,
    # which we can otherwise skip. For the Crazyflie this term can be neglected
    # as the quad's inertia is very small.

    Z91 = np.zeros((9, 1))
    Dx_u = np.block([
        [           Z31, Z33],
        [           Z31, Z33],
        [c.dt * R[:, 2], Z33],
        [           Z91, Z93],
        [           Z31, dt3],
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
# slight loosening of defaults
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

        print("du/dth")
        finitediff_check(th.to_arr(), Du_th, ctrl_th2u, Param.dim_str)

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

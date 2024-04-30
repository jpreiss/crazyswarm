from sympy import *
import numpy as np


def error(R, Rd):
    """Returns error on Lie algebra, plus Jacobians (3 x 9).

    Note this error starts *decreasing* as the angle exceeds 90 degrees, so it
    is nonsensical. Also it has a negative second derivative so it really only
    makes sense for small angles like 45 degrees or less (see [1] for details).

    However, we use it here because its Jacobian is so simple.

    [1] Globally-Attractive Logarithmic Geometric Control of a Quadrotor for
    Aggressive Trajectory Tracking. Jacob Johnson and Randal Beard.
    https://arxiv.org/abs/2109.07025
    """
    errmat = 0.5 * (Rd.T @ R - R.T @ Rd)
    err = np.array([errmat[2, 1], errmat[0, 2], errmat[1, 0]])
    Rx, Ry, Rz = R.T
    Rdx, Rdy, Rdz = Rd.T
    Z = np.zeros((1, 3))
    JR = 0.5 * np.block([
        [     Z,  Rdz.T, -Rdy.T],
        [-Rdz.T,      Z,  Rdx.T],
        [ Rdy.T, -Rdx.T,      Z],
    ])
    JRd = 0.5 * np.block([
        [    Z, -Rz.T,  Ry.T],
        [ Rz.T,     Z, -Rx.T],
        [-Ry.T,  Rx.T,     Z],
    ])
    return err, JR, JRd


def SO3exp(v):
    vn = np.sqrt(np.sum(v ** 2))
    vn1 = 1.0 / vn
    vn2 = vn1 * vn1
    vn3 = vn1 * vn2
    s = np.sin(vn)
    c = np.cos(vn)
    top = -vn1 * s * v
    vvT = np.outer(v, v)
    rest = vn1 * s * np.eye(3) + (c * vn2 - s * vn3) * vvT
    J = np.vstack([top, rest])
    return J


def Jlog_mine(q):
    w = q[0]
    v = q[1:]
    vn = np.sqrt(np.sum(v ** 2))
    vn1 = 1.0 / vn
    vn2 = vn1 * vn1
    rest = np.arccos(w) * (vn * np.eye(3) - vn1 * np.outer(v, v))
    J = vn2 * np.hstack([-v[:, None], rest])
    return J



def evalf(x):
    return np.array(x).astype(np.float64)

def main():
    w, x, y, z = symbols("w x y z", real=True)

    # logarithmic map
    q = Quaternion(w, x, y, z, norm=1)
    log = q._ln().to_Matrix(vector_only=True)
    q = q.to_Matrix()
    Jl = log.jacobian(q)

    # exponential map
    lg = Matrix([x, y, z])
    q = Quaternion(0, *lg).exp().to_Matrix()
    Je = q.jacobian(lg)

    # numerical check
    rng = np.random.default_rng(0)
    for i in range(1):
        # small enough to avoid wrap, otherwise log(exp(imag)) != imag.
        xyz = 0.5 * rng.normal(size=3)
        subs = list(zip([x, y, z], xyz))
        Jen = evalf(Je.subs(subs))

        Jen_mine = Jexp_mine(xyz)
        assert np.allclose(Jen, Jen_mine)

        wxyz = evalf(q.subs(subs)).squeeze()
        Jln = evalf(Jl.subs(zip([w, x, y, z], wxyz)))

        Jln_mine = Jlog_mine(wxyz)
        assert np.allclose(Jln, Jln_mine)

        prod = Jln @ Jen
        if not np.allclose(prod, np.eye(3)):
            print(xyz)
            assert False

    # symbolic results with common subexpression elimination
    v = symbols("v", real=True, positive=True)
    S, C = symbols("S C", real=True)
    sub = [
        (sqrt(x**2 + y**2 + z**2), v),
        (sin(v), S),
        (cos(v), C),
    ]

    Je = Je.subs(sub)
    vJe = (Je).expand()
    print("d exp / d q = (1/v) *")
    pprint(vJe)

    # Je2 = BlockMatrix([
    #     [-S * lg.transpose() / v],
    #     [S / v * eye(3) + (C / v**2 - S / v**3) * lg * lg.transpose()]
    # ]).as_explicit()
    # pprint(Je2)
    # pprint((vJe - Je2).simplify())

    v = symbols("v", real=True, positive=True)
    AC = symbols("AC", real=True)
    Jl = Jl.subs(sub)
    Jl = Jl.subs(1 - w**2, v**2)
    Jl = Jl.subs(acos(w), AC)
    v2Jl = (v**2 * Jl).expand()
    print("d log / d q = (1/v^2) *")
    pprint(v2Jl)


if __name__ == "__main__":
    main()

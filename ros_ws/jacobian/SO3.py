from sympy import *
import numpy as np
import scipy as sp


def hat(w):
    x, y, z = w
    return np.array([
        [ 0, -z,  y],
        [ z,  0, -x],
        [-y,  x,  0],
    ])


def vee(skew):
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def exp(skew):
    return sp.linalg.expm(skew)


def project(X):
    U, _, VT = np.linalg.svd(X)
    return U @ VT


def SO3exp_jac(v):
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


def random(rng, dim=3):
    G = rng.normal(size=(dim, dim))
    U, _, VT = np.linalg.svd(G)
    return U @ VT


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

from collections import namedtuple

import colorama
import numpy as np

import planar
from planar import angleto


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
        dict(to_arr=to_arr, from_arr=from_arr, dim_str=dim_str),
    )


State = namedvec("State", "ierr p v r w", "2 2 2 1 1")
Action = namedvec("Action", "thrust torque", "1 1")
Target = namedvec("Target", "p_d v_d a_d w_d", "2 2 2 1")
Param = namedvec("Param", "ki kp kv kr kw", "1 1 1 1 1")
Const = namedvec("Const", "g m j dt", "1 1 3 1")


def ctrl(x: State, xd: Target, th: Param, c: Const):
    args = [*x] + [*xd] + [*th]
    vals, Du_x, Du_th = planar.ctrl(*args)
    u = Action(*vals)
    return u, Du_x, Du_th


def dynamics(x: State, xd: Target, u: Action, c: Const):
    args = [*x] + [*xd] + [*u] + [c.dt,]
    vals, Dx_x, Dx_u = planar.dynamics(*args)
    xt = State(*vals)
    return xt, Dx_x, Dx_u


# TODO: costs?


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
        x = State.from_arr(rng.normal(size=8))
        xd = Target.from_arr(rng.normal(size=7))
        th = Param.from_arr(rng.uniform(0.1, 4, size=5))

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

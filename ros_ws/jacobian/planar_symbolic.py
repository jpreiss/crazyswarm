import sympy as sym


# Can't use MatrixSymbol because there's no way to declare realness.
def vec2symbols(names):
    names = names.split(" ")
    out = []
    for name in names:
        nx, ny = sym.symbols(f"{name}_x {name}_y", real=True)
        out.append(sym.Matrix([nx, ny]))
    return out


def angleto(a, b):
    ax, ay = a
    R = sym.Matrix([
        [ ax, ay],
        [-ay, ax],
    ])
    bprime = R @ b
    #angle = sym.atan2(bprime[1], bprime[0])
    # NOTE: works for 90 degrees or less
    angle = sym.asin(bprime[1])
    return angle


def main():
    # INPUTS
    # ------
    # states and targets
    ierr, p, v, p_d, v_d, a_d = vec2symbols("ierr p v p_d v_d a_d")
    r, w, w_d = sym.symbols("r w w_d", real=True)
    x = sym.Matrix([*ierr, *p, *v, r, w])

    # params
    ki, kp, kv, kr, kw = sym.symbols("ki kp kv kr kw", real=True, positive=True)
    theta = sym.Matrix([ki, kp, kv, kr, kw])

    # constants (NOTE: mass/inertia could be params!)
    g, m, j, dt = sym.symbols("g m j dt", real=True, positive=True)
    g = sym.Matrix([0, g])

    # derived states
    right = sym.Matrix([sym.cos(r), sym.sin(r)])
    up = sym.Matrix([-sym.sin(r), sym.cos(r)])

    # CONTROLLER
    # ----------
    # position part
    feedback = -ki * ierr - kp * (p - p_d) - kv * (v - v_d)
    tvec_actual = feedback + a_d + g
    # TODO: compare perf with and without projection!
    tnorm_actual = tvec_actual.norm()
    #ulin = sym.symbols("ulin", real=True, positive=True)
    tnorm = sym.symbols("tnorm", real=True, positive=True)
    tvec, = vec2symbols("tvec")

    # attitude part
    er_actual = angleto(tvec / tnorm, up)
    #er = sym.symbols("er", real=True)
    er = er_actual
    urot = -kr * er - kw * (w_d - w)
    uactual = sym.Matrix([tnorm, urot])
    u, = vec2symbols("u")

    # DYNAMICS
    # --------

    # Normally I would use symplectic Euler integration, but plain forward
    # Euler gives simpler Jacobians.

    # attitude part
    w_t = w + dt * u[1]
    r_t = r + dt * w

    # position part
    a = u[0] * up - g
    v_t = v + dt * a
    p_t = p + dt * v
    ierr_t = ierr + dt * (p - p_d)
    x_t = sym.Matrix([*ierr_t, *p_t, *v_t, r_t, w_t])

    # JACOBIANS
    def printclean(val):
        sym.pprint(val)

    print("dxdx =")
    dxdx = x_t.jacobian(x)
    printclean(dxdx)

    print("dxdu =")
    dxdu = x_t.jacobian(u)
    printclean(dxdu)

    print("dtnormdx =")
    printclean(sym.Matrix([tnorm_actual]).jacobian(x))

    print("dudx =")
    dudx = uactual.jacobian(x)
    printclean(dudx)

    print("dudtnorm =")
    printclean(uactual.jacobian(sym.Matrix([tnorm])))

    print("dudth =")
    dudth = uactual.jacobian(theta)
    printclean(dudth)




if __name__ == "__main__":
    main()

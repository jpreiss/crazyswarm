import sympy as sym

def main():
    a, b, c, d, t, T, X = sym.symbols("a,b,c,d,t,T,X", real=True)
    p = a * t**3 + b * t**2 + c * t + d
    p = p.as_poly()
    pdot = sym.diff(p, t)
    eqns = [
        p.subs(t, 0),
        pdot.subs(t, 0),
        p.subs(t, T) - X,
        pdot.subs(t, T) - 1,
    ]
    soln = sym.solve(eqns, [a, b, c, d])
    sym.pprint(soln)


if __name__ == "__main__":
    main()


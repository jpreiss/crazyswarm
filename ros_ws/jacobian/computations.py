import sympy

# compound inputs

# TODO: how to flatten?
def vec(M):
    rows, cols = M.shape
    elts = [*M.transpose()]
    return sym.Matrix(elts)

Rflat = vec(R)
X = sym.Matrix.vstack(i_error_pos, statePos, stateVel, Rflat, omega)
# TODO: mass and inertia too?
Theta = sym.Matrix.vstack(Kpos_I, Kpos_P, Kpos_D, KI, KR, Komega)

u : sympy.MatAdd
u = u.as_explicit()
dudx = u.jacobian(X)
print("du/dx =")
sym.pprint(dudx)

dudth = u.jacobian(Theta)
print("du/dtheta =")
sym.pprint(dudtheta)


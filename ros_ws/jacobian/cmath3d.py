"""
manual steps I did:
- eliminate `if` statements for unused paths and special cases
- collapse multi-line statements to one line
- remove some dead code
"""

import re
import sys
import warnings

import sympy as sym


# quat
def mkquat(x, y, z, w):
    return sym.algebras.Quaternion(w, x, y, z)

def quat2rotmat(q):
    return q.to_rotation_matrix()

def quat2rpy(q):
    # copied from cmath3d.h, too many different ways to do it
    r = sym.atan2(2.0 * (q.a * q.b + q.c * q.d), 1 - 2 * (fsqr(q.b) + fsqr(q.c)))
    p = sym.asin(2.0 * (q.a * q.c - q.b * q.d))
    y = sym.atan2(2.0 * (q.a * q.d + q.b * q.c), 1 - 2 * (fsqr(q.c) + fsqr(q.d)))
    return mkvec(r, p, y)

def mat2quat(m):
    return sym.algebras.Quaternion.from_rotation_matrix(m)

# matrix
def mmul(a, b):
    return a @ b

def mvmul(A, x):
    return A @ x

def mcolumn(A, i):
    return A[:, i]

def mcolumns(a, b, c):
    return sym.BlockMatrix([[a.T], [b.T], [c.T]]).T

def msub(A, B):
    return A - B

def mtranspose(A):
    return A.transpose()


# vector
def mkvec(x, y, z):
    return sym.Matrix([x, y, z])

def vadd(a, b):
    return a + b

def vadd3(a, b, c):
    return a + b + c

def vadd4(a, b, c, d):
    return a + b + c + d

def vsub(a, b):
    return a - b

def vsub2(a, b, c):
    return a - b - c

def veltmul(a, b):
    return sym.HadamardProduct(a, b)

def vscl(s, a):
    return s * a

def vbasis(i):
    vals = [0, 0, 0]
    vals[i] = 1
    return sym.Matrix(vals)

def vdot(a, b):
    # sympy's dot is weird about complex matrix expressions
    return sum(a[i, 0] * b[i, 0] for i in range(3))

def vcross(a, b):
    try:
        return a.cross(b[:, 0])
    except:
        return -b.cross(a[:, 0])

def vzero():
    return sym.Matrix([0, 0, 0])

def vneg(a):
    return -a

def vmag(a):
    return sym.sqrt(vdot(a, a))

def vnormalize(a):
    return a / vmag(a)

def vclampscl(a, lb, ub):
    # not differentiable -- ignore!!!
    return a

# scalar
def cosf(x):
    return sym.cos(x)

def sinf(x):
    return sym.sin(x)

def radians(x):
    return x

def fsqr(x):
    return x ** 2

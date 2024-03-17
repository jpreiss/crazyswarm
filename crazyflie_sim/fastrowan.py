import sys
from types import ModuleType

from cffirmware import *
import numpy as np


def a2q(a):
    return mkquat(a[1], a[2], a[3], a[0])


def q2a(q):
    return np.array([q.w, q.x, q.y, q.z])


def rotate(q, v):
    fq = a2q(q)
    fv = mkvec(*v)
    return np.array(qvrot(fq, fv))


def normalize(q):
    return q2a(qnormalize(a2q(q)))


def to_euler(q, convention):
    assert convention == "xyz"
    return np.array(quat2rpy(a2q(q)))


def from_matrix(R):
    Rm = mrows(*[mkvec(row) for row in R])
    return q2a(mat2quat(Rm))


class calculus:
    @staticmethod
    def integrate(q, omega, dt):
        return q2a(quat_gyro_update(a2q(q), mkvec(*omega), dt))

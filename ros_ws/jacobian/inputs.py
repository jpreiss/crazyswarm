from cmath3d import *

from collections import namedtuple

class XYZW:
    def __init__(self, name):
        for char in "xyzw":
            setattr(self, char, sym.Symbol(f"{name}_{char}", real=True))

class RPY:
    def __init__(self, name):
        for elt in ["roll", "pitch", "yaw"]:
            setattr(self, elt, sym.Symbol(f"{name}_{elt}", real=True))

setpoint_position = XYZW("setpoint_position")
setpoint_velocity = XYZW("setpoint_velocity")
setpoint_acceleration = XYZW("setpoint_acceleration")
setpoint_jerk = XYZW("setpoint_jerk")
setpoint_attitudeRate = RPY("setpoint_attitudeRate")
state_position = XYZW("state_position")
state_velocity = XYZW("state_velocity")
sensors_gyro = XYZW("sensors_gyro")
#state_attitudeQuaternion = XYZW("state_attitudeQuaternion")
#state_acceleration = XYZ("state_acceleration")
R = sym.MatrixSymbol("R", 3, 3)

desiredYaw = sym.Symbol("desiredYaw", real=True)

# in code but not used
Kpos_P_limit = 0
Kpos_D_limit = 0

def symvec(name):
    v = XYZW(name)
    return sym.Matrix([v.x, v.y, v.z])

# params
Kpos_P = symvec("Kpos_P")
Kpos_I = symvec("Kpos_I")
Kpos_D = symvec("Kpos_D")
KR = symvec("KR")
KI = symvec("KI")
J = symvec("J")
Komega = symvec("Komega")
omega = symvec("omega")
mass = sym.Symbol("mass", real=True, positive=True)
dt = sym.Symbol("dt", real=True, positive=True)

i_error_pos = symvec("i_error_pos")
i_error_att = symvec("i_error_att")

GRAVITY_MAGNITUDE = sym.Symbol("g", real=True, positive=True)

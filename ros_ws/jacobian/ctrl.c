// struct vec dessnap = vzero();
// Address inconsistency in firmware where we need to compute our own desired yaw angle
// Rate-controlled YAW is moving YAW angle setpoint

// Position controller
struct vec pos_d = mkvec(setpoint->position.x, setpoint->position.y, setpoint->position.z);
struct vec vel_d = mkvec(setpoint->velocity.x, setpoint->velocity.y, setpoint->velocity.z);
struct vec acc_d = mkvec(setpoint->acceleration.x, setpoint->acceleration.y, setpoint->acceleration.z + GRAVITY_MAGNITUDE);
struct vec statePos = mkvec(state->position.x, state->position.y, state->position.z);
struct vec stateVel = mkvec(state->velocity.x, state->velocity.y, state->velocity.z);

// errors
struct vec pos_e = vclampscl(vsub(pos_d, statePos), -self->Kpos_P_limit, self->Kpos_P_limit);
struct vec vel_e = vclampscl(vsub(vel_d, stateVel), -self->Kpos_D_limit, self->Kpos_D_limit);
// NOTE: leaving out because we will do this in the dynamics model
//self->i_error_pos = vadd(self->i_error_pos, vscl(dt, pos_e));
self->p_error = pos_e;
self->v_error = vel_e;

struct vec F_d = vadd4( acc_d, veltmul(self->Kpos_D, vel_e), veltmul(self->Kpos_P, pos_e), veltmul(self->Kpos_I, self->i_error_pos));

//struct quat q = mkquat(state->attitudeQuaternion.x, state->attitudeQuaternion.y, state->attitudeQuaternion.z, state->attitudeQuaternion.w);
//struct mat33 R = quat2rotmat(q);
//struct vec z	= vbasis(2);
control->thrustSi = self->mass*vdot(F_d , mcolumn(R, 2));
self->thrustSi = control->thrustSi;

// Compute Desired Rotation matrix
float normFd = control->thrustSi;

zdes = vnormalize(F_d);
struct vec xcdes = mkvec(cosf(desiredYaw), sinf(desiredYaw), 0); 
struct vec zcrossx = vcross(zdes, xcdes);
float normZX = vmag(zcrossx);

ydes = vnormalize(zcrossx);
xdes = vcross(ydes, zdes);

self->R_des = mcolumns(xdes, ydes, zdes);

// Attitude controller

// current rotation [R]
// NOTE: using quats was giving us overly complex symbolic math
// struct quat q = mkquat(state->attitudeQuaternion.x, state->attitudeQuaternion.y, state->attitudeQuaternion.z, state->attitudeQuaternion.w);
// struct mat33 R = quat2rotmat(q);

// rotation error
struct mat33 eRM = msub(mmul(mtranspose(self->R_des), R), mmul(mtranspose(R), self->R_des));

struct vec eR = vscl(0.5f, mkvec(eRM.m[2][1], eRM.m[0][2], eRM.m[1][0]));

// angular velocity
self->omega = mkvec( radians(sensors->gyro.x), radians(sensors->gyro.y), radians(sensors->gyro.z));

// Compute desired omega
struct vec xdes = mcolumn(self->R_des, 0);
struct vec ydes = mcolumn(self->R_des, 1);
struct vec zdes = mcolumn(self->R_des, 2);
struct vec hw = vzero();
// Desired Jerk and snap for now are zeros vector
struct vec desJerk = mkvec(setpoint->jerk.x, setpoint->jerk.y, setpoint->jerk.z);

struct vec tmp = vsub(desJerk, vscl(vdot(zdes, desJerk), zdes));
hw = vscl(self->mass/control->thrustSi, tmp);
struct vec z_w = mkvec(0,0,1); 
float desiredYawRate = radians(setpoint->attitudeRate.yaw) * vdot(zdes,z_w);
struct vec omega_des = mkvec(-vdot(hw,ydes), vdot(hw,xdes), desiredYawRate);

self->omega_r = mvmul(mmul(mtranspose(R), self->R_des), omega_des);

struct vec omega_error = vsub(self->omega, self->omega_r);

// Integral part on angle
self->i_error_att = vadd(self->i_error_att, vscl(dt, eR));

// compute moments
// M = -kR eR - kw ew + w x Jw - J(w x wr)
self->u = vadd4( vneg(veltmul(self->KR, eR)), vneg(veltmul(self->Komega, omega_error)), vneg(veltmul(self->KI, self->i_error_att)), vcross(self->omega, veltmul(self->J, self->omega)));

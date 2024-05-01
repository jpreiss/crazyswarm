#include <cmath>
#include <tuple>

#include <Eigen/Dense>
#include <Eigen/KroneckerProduct>


// NOTE: FLOAT must resolve to a floating-point type. Add a `using` before
// including this header, i.e.
//
// using FLOAT = double;
// #include "gapsquad.hpp"

// externally visible types / constants
int constexpr XDIM = 3 + 3 + 3 + 9 + 3;
int constexpr UDIM = 1 + 3;
int constexpr TDIM = 2 * 5; // 5 params, each with xy and z variants

using Vec = Eigen::Matrix<FLOAT, 3, 1>;
using Mat = Eigen::Matrix<FLOAT, 3, 3, Eigen::RowMajor>;
using Jxx = Eigen::Matrix<FLOAT, XDIM, XDIM, Eigen::RowMajor>;
using Jxu = Eigen::Matrix<FLOAT, XDIM, UDIM, Eigen::RowMajor>;
using Jut = Eigen::Matrix<FLOAT, UDIM, TDIM, Eigen::RowMajor>;
using Jux = Eigen::Matrix<FLOAT, UDIM, XDIM, Eigen::RowMajor>;

using StateTuple = std::tuple<Vec, Vec, Vec, Mat, Vec>;
using ActionTuple = std::tuple<FLOAT, Vec>;
using TargetTuple = std::tuple<Vec, Vec, Vec, FLOAT, Vec>;
using ParamTuple = std::tuple<
	FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, // position gains
	FLOAT, FLOAT, FLOAT, FLOAT // attitude gains
>;

struct State { Vec ierr; Vec p; Vec v; Mat R; Vec w; };
struct Action { FLOAT thrust; Vec torque; };
struct Target { Vec p_d; Vec v_d; Vec a_d; FLOAT y_d; Vec w_d; };
struct Param {
	FLOAT ki_xy; FLOAT ki_z; FLOAT kp_xy; FLOAT kp_z; FLOAT kv_xy; FLOAT kv_z; // position gains
	FLOAT kr_xy; FLOAT kr_z; FLOAT kw_xy; FLOAT kw_z; // attitude gains
};

// The C++ standard doesn't guarantee that this will hold, but it does in
// practice (at least on Clang++ 15), so we can use it to catch errors due to
// adding/removing a field.
static_assert(sizeof(StateTuple) == sizeof(State), "struct/tuple error");
static_assert(sizeof(ActionTuple) == sizeof(Action), "struct/tuple error");
static_assert(sizeof(TargetTuple) == sizeof(Target), "struct/tuple error");
static_assert(sizeof(ParamTuple) == sizeof(Param), "struct/tuple error");

// internally visible types / constants
FLOAT constexpr GRAV = 9.81;
using Mat39 = Eigen::Matrix<FLOAT, 3, 9>;
using Mat93 = Eigen::Matrix<FLOAT, 9, 3>;
using Mat99 = Eigen::Matrix<FLOAT, 9, 9>;
using VecT = Eigen::Matrix<FLOAT, 1, 3>;

std::tuple<Vec, Vec, Vec> colsplit(Mat const &m)
{
	return std::make_tuple(m.col(0), m.col(1), m.col(2));
}

Mat fromcols(Vec const &a, Vec const &b, Vec const &c)
{
	Mat m;
	m.col(0) = a;
	m.col(1) = b;
	m.col(2) = c;
	return m;
}


/*
Returns error on Lie algebra, plus Jacobians (3 x 9).

Note this error starts *decreasing* as the angle exceeds 90 degrees, so it is
nonsensical. Also it has a negative second derivative so it really only makes
sense for small angles like 45 degrees or less (see [1] for details).

However, we use it here because its Jacobian is so simple.

[1] Globally-Attractive Logarithmic Geometric Control of a Quadrotor for
Aggressive Trajectory Tracking. Jacob Johnson and Randal Beard.
https://arxiv.org/abs/2109.07025
*/
std::tuple<Vec, Mat39, Mat39> SO3error(Mat const &R, Mat const &Rd)
{
	Mat errmat = 0.5 * (Rd.transpose() * R - R.transpose() * Rd);
	Vec err {errmat(2, 1), errmat(0, 2), errmat(1, 0)};
	Vec Rx, Ry, Rz;
	std::tie(Rx, Ry, Rz) = colsplit(R);
	Vec Rdx, Rdy, Rdz;
	std::tie(Rdx, Rdy, Rdz) = colsplit(Rd);
	VecT Z = VecT::Zero();
	Mat39 JR = 0.5 * (Mat39() <<
		               Z,  Rdz.transpose(), -Rdy.transpose(),
		-Rdz.transpose(),                Z,  Rdx.transpose(),
		 Rdy.transpose(), -Rdx.transpose(),               Z).finished();
	Mat39 JRd = 0.5 * (Mat39() <<
		              Z, -Rz.transpose(),  Ry.transpose(),
		 Rz.transpose(),               Z, -Rx.transpose(),
		-Ry.transpose(),  Rx.transpose(),              Z).finished();
	// these are the signs that make torque = -k * error work for k > 0.
	return std::make_tuple(-err, -JR, -JRd);
}

Mat hat(Vec const &w)
{
	FLOAT x = w[0], y = w[1], z = w[2];
	Mat m = (Mat() <<
		 0, -z,  y,
		 z,  0, -x,
		-y,  x,  0).finished();
	return m;
}

std::tuple<Vec, Mat> normalize(Vec const &v)
{
	FLOAT vn = 1.0 / v.norm();
	FLOAT vn3 = vn * vn * vn;
	Mat J = vn * Mat::Identity() - vn3 * v * v.transpose();
	return std::make_tuple(vn * v, J);
}

std::tuple<Vec, Mat, Mat> cross(Vec const &a, Vec const &b)
{
	FLOAT ax = a[0], ay = a[1], az = a[2];
	FLOAT bx = b[0], by = b[1], bz = b[2];
	Vec x {
		ay * bz - az * by,
		az * bx - ax * bz,
		ax * by - ay * bx
	};
	Mat Ja = -hat(b);
	Mat Jb = hat(a);
	return std::make_tuple(x, Ja, Jb);
}


void dynamics(
	Vec const &ierr, Vec const &p, Vec const &v, Mat const &R, Vec const &w, // state
	Vec const &p_d, Vec const &v_d, Vec const &a_d, FLOAT y_d, Vec const &w_d, // target
	FLOAT thrust, Vec const &torque, // input
	FLOAT dt, // constants
	State &x_t, Jxx &Dx_x, Jxu &Dx_u // outputs
	)
{
	Vec g {0, 0, GRAV};
	Mat I3 = Mat::Identity();
	Mat99 I9 = Mat99::Identity();

	Vec up = R.col(2);
	Vec acc = thrust * up - g;

	Eigen::Matrix<FLOAT, 3, XDIM> Dacc_x;
	Dacc_x.setZero();
	// I3 wrt Z column of R
	Dacc_x.block<3, 3>(0, 3 + 3 + 3 + 6) = thrust * I3;

	// Normally I would use symplectic Euler integration, but plain forward
	// Euler gives simpler Jacobians.

	x_t.ierr = ierr + dt * (p - p_d);
	x_t.p = p + dt * v;
	x_t.v = v + dt * acc;
	x_t.R = R + dt * R * hat(w);
	x_t.w = w + dt * torque;

	// TODO: This became trivial after we went from angle state to rotation
	// matrix -- condense some ops.
	Mat39 Dvt_R = dt * Dacc_x.block<3, 9>(0, 9);

	Mat99 DRt_R = I9 + dt * kroneckerProduct(hat(-w), I3);

	Vec Rx, Ry, Rz;
	std::tie(Rx, Ry, Rz) = colsplit(R);

	Mat93 DRt_w;
	// shift operator constructor transposes everything by itself!
	// VecT Z31 = Vec::Zero();
	// DRt_w <<
		// Z31, -Rz,  Ry,
		 // Rz, Z31, -Rx,
		// -Ry,  Rx, Z31;
	DRt_w.setZero();
	/* 0 */                                DRt_w.block<3, 1>(0, 1) = -dt * Rz;    DRt_w.block<3, 1>(0, 2) =  dt * Ry;
	DRt_w.block<3, 1>(3, 0) =  dt * Rz;    /* 0 */                                DRt_w.block<3, 1>(3, 2) = -dt * Rx;
	DRt_w.block<3, 1>(6, 0) = -dt * Ry;    DRt_w.block<3, 1>(6, 1) =  dt * Rx;    /* 0 */

	// auto keeps the expression templates, for possible optimization
	auto Z33 = Mat::Zero();
	auto Z39 = Mat39::Zero();
	auto Z93 = Mat93::Zero();
	// auto Z91 = Eigen::Matrix<FLOAT, 9, 1, Eigen::RowMajor>::Zero();
	auto dt3 = dt * I3;

	Dx_x <<
		 I3, dt3, Z33,   Z39,   Z33,
		Z33,  I3, dt3,   Z39,   Z33,
		Z33, Z33,  I3, Dvt_R,   Z33,
		Z93, Z93, Z93, DRt_R, DRt_w,
		Z33, Z33, Z33,   Z39,    I3;
	// (Refers to Dx_x construction above.) Skipping Coriolis term that would
	// make dw'/dw nonzero because it requires system ID of the inertia matrix,
	// which we can otherwise skip. For the Crazyflie this term can be
	// neglected as the quad's inertia is very small.

	Dx_u.setZero();
	Dx_u.block<3, 1>(6, 0) = dt * Rz;
	Dx_u.block<3, 3>(9 + 9, 1) = dt3;
}

template <typename S, typename T>
bool allclose(S &&s, T &&t, FLOAT atol=1e-8, FLOAT rtol=1e-5)
{
	auto a = (s - t).array().abs();
	auto allowed = rtol * t.array().abs() + atol;
	return (a < allowed).all();
}

void ctrl(
	Vec ierr, Vec p, Vec v, Mat R, Vec w, // state
	Vec p_d, Vec v_d, Vec a_d, FLOAT y_d, Vec w_d, // target
	FLOAT ki_xy, FLOAT ki_z, FLOAT kp_xy, FLOAT kp_z, FLOAT kv_xy, FLOAT kv_z, // position gains
	FLOAT kr_xy, FLOAT kr_z, FLOAT kw_xy, FLOAT kw_z, // attitude gains
	Action &u, Jux &Du_x, Jut &Du_th // outputs
	)
{
	Vec g {0, 0, GRAV};
	Mat I = Mat::Identity();

	using Diag = Eigen::DiagonalMatrix<FLOAT, 3>;
	Diag ki {ki_xy, ki_xy, ki_z};
	Diag kp {kp_xy, kp_xy, kp_z};
	Diag kv {kv_xy, kv_xy, kv_z};
	Diag kr {kr_xy, kr_xy, kr_z};
	Diag kw {kw_xy, kw_xy, kw_z};

	// position part components
	Vec perr = p - p_d;
	Vec verr = v - v_d;
	// Parens because Eigen forbids negating diagonal matrices for some reason?
	Vec feedback = - (ki * ierr) - (kp * perr) - (kv * verr);
	Vec a = feedback + a_d + g;

	Eigen::Matrix<FLOAT, 3, XDIM> Da_x;
	Da_x << -(ki * I), -(kp * I), -(kv * I), Eigen::Matrix<FLOAT, 3, 9 + 3>::Zero();

	Eigen::Matrix<FLOAT, 3, TDIM> Da_th;
	Da_th <<
		-ierr[0],        0, -perr[0],        0, -verr[0],        0, 0, 0, 0, 0,
		-ierr[1],        0, -perr[1],        0, -verr[1],        0, 0, 0, 0, 0,
		       0, -ierr[2],        0, -perr[2],        0, -verr[2], 0, 0, 0, 0;

	u.thrust = a.norm();
	VecT Dthrust_a = (a / u.thrust).transpose();

	Vec zgoal;
	Mat Dzgoal_a;
	std::tie(zgoal, Dzgoal_a) = normalize(a);

	Vec xgoalflat { std::cos(y_d), std::sin(y_d), 0 };
	Vec ygoalnn;
	Mat Dygoalnn_zgoal, dummy;
	std::tie(ygoalnn, Dygoalnn_zgoal, dummy) = cross(zgoal, xgoalflat);
	Vec ygoal;
	Mat Dygoal_ygoalnn;
	std::tie(ygoal, Dygoal_ygoalnn) = normalize(ygoalnn);
	Mat Dygoal_a = Dygoal_ygoalnn * Dygoalnn_zgoal * Dzgoal_a;

	Vec xgoal;
	Mat Dxgoal_ygoal, Dxgoal_zgoal;
	std::tie(xgoal, Dxgoal_ygoal, Dxgoal_zgoal) = cross(ygoal, zgoal);
	Mat Dxgoal_a = Dxgoal_ygoal * Dygoal_a + Dxgoal_zgoal * Dzgoal_a;
	Mat Rd = fromcols(xgoal, ygoal, zgoal);

	#ifndef CRAZYFLIE_FW
	{
		// extra correctness checks
		if (!allclose(zgoal, a / u.thrust)) {
			throw std::runtime_error("normalization wrong");
		}
		FLOAT norm = xgoal.norm();
		if (std::abs(norm - 1) > 1e-7) {
			throw std::runtime_error("xgoal norm too far from 1: is " + std::to_string(norm));
		}
		FLOAT det = Rd.determinant();
		if (std::abs(det - 1) > 1e-7) {
			throw std::runtime_error("Rd determinant too far from 1: is " + std::to_string(det));
		}
		Mat RdTRd = Rd.transpose() * Rd;
		FLOAT maxerr = (RdTRd - I).array().abs().maxCoeff();
		if (maxerr > 1e-7) {
			throw std::runtime_error("Rd is not orthogonal: maxerr is " + std::to_string(maxerr));
		}
	}
	#endif

	Eigen::Matrix<FLOAT, 9, 3> DRd_a;
	DRd_a.block<3, 3>(0, 0) = Dxgoal_a;
	DRd_a.block<3, 3>(3, 0) = Dygoal_a;
	DRd_a.block<3, 3>(6, 0) = Dzgoal_a;

	Vec er;
	Mat39 Der_R, Der_Rd;
	std::tie(er, Der_R, Der_Rd) = SO3error(R, Rd);

	Vec ew = w - w_d;
	u.torque = -(kr * er) - (kw * ew);

	// controller chain rules
	auto Dthrust_x = Dthrust_a * Da_x;
	auto Dthrust_th = Dthrust_a * Da_th;

	Eigen::Matrix<FLOAT, 3, XDIM> Der_x;
	Der_x.setZero();
	Der_x.block<3, 9>(0, 9) = Der_R;
	Der_x += Der_Rd * DRd_a * Da_x;

	Eigen::Matrix<FLOAT, 3, TDIM> Der_th = Der_Rd * DRd_a * Da_th;

	Eigen::Matrix<FLOAT, 3, XDIM> Dtorque_x = -(kr * Der_x);
	Dtorque_x.block<3, 3>(0, 3 + 3 + 3 + 9) -= kw * I;

	Eigen::Matrix<FLOAT, 3, TDIM> Dtorque_th = -(kr * Der_th); // indirect part
	Dtorque_th += (Eigen::Matrix<FLOAT, 3, TDIM>() <<
		0, 0, 0, 0, 0, 0, -er[0],      0, -ew[0],      0,
		0, 0, 0, 0, 0, 0, -er[1],      0, -ew[1],      0,
		0, 0, 0, 0, 0, 0,      0, -er[2],      0, -ew[2]).finished();

	Du_x << Dthrust_x, Dtorque_x;
	Du_th << Dthrust_th, Dtorque_th;
}

std::tuple<ActionTuple, Jux, Jut>
ctrl_wrap(
	Vec ierr, Vec p, Vec v, Mat R, Vec w, // state
	Vec p_d, Vec v_d, Vec a_d, FLOAT y_d, Vec w_d, // target
	FLOAT ki_xy, FLOAT ki_z, FLOAT kp_xy, FLOAT kp_z, FLOAT kv_xy, FLOAT kv_z, // position gains
	FLOAT kr_xy, FLOAT kr_z, FLOAT kw_xy, FLOAT kw_z // attitude gains
	)
{
	std::tuple<ActionTuple, Jux, Jut> output;
	Action a;
	ctrl(
		ierr, p, v, R, w,
		p_d, v_d, a_d, y_d, w_d,
		ki_xy, ki_z, kp_xy, kp_z, kv_xy, kv_z,
		kr_xy, kr_z, kw_xy, kw_z,
		a, std::get<Jux>(output), std::get<Jut>(output)
	);
	std::get<ActionTuple>(output) = std::make_tuple(a.thrust, a.torque);
	return output;
}

std::tuple<StateTuple, Jxx, Jxu>
dynamics_wrap(
	Vec const &ierr, Vec const &p, Vec const &v, Mat const &R, Vec const &w, // state
	Vec const &p_d, Vec const &v_d, Vec const &a_d, FLOAT y_d, Vec const &w_d, // target
	FLOAT thrust, Vec const &torque, // input
	FLOAT dt // constants
	)
{
	std::tuple<StateTuple, Jxx, Jxu> output;
	State s;
	dynamics(
		ierr, p, v, R, w,
		p_d, v_d, a_d, y_d, w_d,
		thrust, torque,
		dt,
		s, std::get<Jxx>(output), std::get<Jxu>(output)
	);
	std::get<StateTuple>(output) = std::make_tuple(s.ierr, s.p, s.v, s.R, s.w);
	return output;
}

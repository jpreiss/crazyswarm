#include <cmath>
#include <tuple>

#include <Eigen/Dense>


using FLOAT = double;
int constexpr XDIM = 3 + 3 + 3 + 9 + 3;
int constexpr UDIM = 1 + 3;
int constexpr TDIM = 5;
FLOAT constexpr GRAV = 9.81;

using Vec = Eigen::Matrix<FLOAT, 3, 1>;
using VecT = Eigen::Matrix<FLOAT, 1, 3>;
using Mat = Eigen::Matrix<FLOAT, 3, 3, Eigen::RowMajor>;
using Mat39 = Eigen::Matrix<FLOAT, 3, 9, Eigen::RowMajor>;
//using Mat99 = Eigen::Matrix<FLOAT, 9, 9, Eigen::RowMajor>;

using Jxx = Eigen::Matrix<FLOAT, XDIM, XDIM, Eigen::RowMajor>;
using Jxu = Eigen::Matrix<FLOAT, XDIM, UDIM, Eigen::RowMajor>;
using Jut = Eigen::Matrix<FLOAT, UDIM, TDIM, Eigen::RowMajor>;
using Jux = Eigen::Matrix<FLOAT, UDIM, XDIM, Eigen::RowMajor>;

using State = std::tuple<Vec, Vec, Vec, Mat, Vec>;
using Action = std::tuple<FLOAT, Vec>;


std::tuple<Vec, Vec, Vec> colsplit(Mat const &m)
{
	return std::make_tuple(m.col(0), m.col(1), m.col(2));
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
	Mat39 JR, JRd;
	JR <<
		               Z,  Rdz.transpose(), -Rdy.transpose(),
		-Rdz.transpose(),                Z,  Rdx.transpose(),
		 Rdy.transpose(), -Rdx.transpose(),               Z;
	JR *= 0.5;
	JRd << 
		              Z, -Rz.transpose(),  Ry.transpose(),
		 Rz.transpose(),               Z, -Rx.transpose(),
		-Ry.transpose(),  Rx.transpose(),              Z;
	JRd *= 0.5;
	return std::make_tuple(-err, -JR, -JRd);
}

Mat hat(Vec const &w)
{
	FLOAT x = w[0], y = w[1], z = w[2];
	Mat m;
	m << 0, -z,  y,
		 z,  0, -x,
		-y,  x,  0;
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






/*
// Returns angle rotating vec. a to vec. b and gradients.
// Vectors must be unit length.
// Returns: angle, dangle/da, dangle/db
std::tuple<FLOAT, Vec, Vec> angleto(Vec a, Vec b)
{
	// Computes atan2 of b in the coordinate frame of a.
	FLOAT ax = a[0];
	FLOAT ay = a[1];
	FLOAT bx = b[0];
	FLOAT by = b[1];
	Mat R;
	R <<
		 ax, ay,
		-ay, ax;
	Vec Rb = R * b;
	Mat DRb_a;
	DRb_a <<
		bx,  by,
		by, -bx;
	FLOAT angle = std::atan2(Rb[1], Rb[0]);
	Vec grad_atan2 {-Rb[1], Rb[0]};
	Vec grad_a = grad_atan2.transpose() * DRb_a;
	Vec grad_b = grad_atan2.transpose() * R;

	return std::make_tuple(angle, grad_a, grad_b);
}

std::tuple<State, Jxx, Jxu>
dynamics(
	Vec ierr, Vec p, Vec v, FLOAT r, FLOAT w, // state
	Vec p_d, Vec v_d, Vec a_d, FLOAT w_d, // target
	FLOAT thrust, FLOAT torque, // input
	FLOAT dt)
{
	// derived state - TODO factor out?
	Vec up {-std::sin(r), std::cos(r)};
	Eigen::Matrix<FLOAT, 2, XDIM> Dup_x;
	Dup_x <<
		0, 0, 0, 0, 0, 0, -std::cos(r), 0,
		0, 0, 0, 0, 0, 0, -std::sin(r), 0;
	Vec g {0, GRAV};

	// Normally I would use symplectic Euler integration, but plain forward
	// Euler gives simpler Jacobians.

	Vec acc = thrust * up - g;
	Eigen::Matrix<FLOAT, 2, XDIM> Dacc_x = thrust * Dup_x;
	State x_t = std::make_tuple(
		ierr + dt * (p - p_d),
		p + dt * v,
		v + dt * acc,
		r + dt * w,
		w + dt * torque
	);

	Eigen::Matrix<FLOAT, 2, 1> Dvt_r = dt * Dacc_x.col(6);
	Mat I2 = Mat::Identity();

	// dx/dx Jacobian
	Jxx Dx_x = Jxx::Identity();
	// integrators
	Dx_x.block<2, 2>(0, 2) = dt * I2;
	Dx_x.block<2, 2>(2, 4) = dt * I2;
	// angular integrator
	Dx_x(6, 7) = dt;
	// vel-thrust
	Dx_x.block<2, 1>(4, 6) = Dvt_r;

	// dx/du Jacobian
	Jxu Dx_u = Jxu::Zero();
	Dx_u.block<2, 1>(4, 0) = dt * up;
	Dx_u(7, 1) = dt;

	return std::make_tuple(x_t, Dx_x, Dx_u);
}


std::tuple<Action, Jux, Jut>
ctrl(
	Vec ierr, Vec p, Vec v, FLOAT r, FLOAT w, // state
	Vec p_d, Vec v_d, Vec a_d, FLOAT w_d, // target
	FLOAT ki, FLOAT kp, FLOAT kv, FLOAT kr, FLOAT kw // params
	)
{
	// derived state
	Vec up {-std::sin(r), std::cos(r)};
	Vec Dup_r {-std::cos(r), -std::sin(r)};
	Vec g {0, GRAV};

	// position part components
	Vec perr = p - p_d;
	Vec verr = v - v_d;
	Vec feedback = - ki * ierr - kp * perr - kv * verr;
	Vec a = feedback + a_d + g;

	Mat I = Mat::Identity();
	Eigen::Matrix<FLOAT, 2, XDIM> Da_x;
	Da_x << -ki * I, -kp * I, -kv * I, Mat::Zero();

	Eigen::Matrix<FLOAT, 2, TDIM> Da_th;
	Da_th << -ierr, -perr, -verr, Mat::Zero();

	FLOAT thrust = a.norm();
	Vec Dthrust_a = a / thrust;

	Vec upgoal = a / thrust;
	Mat aaT = a * a.transpose();
	Mat Dupgoal_a = (1.0 / thrust) * Mat::Identity() - (1 / (thrust * thrust * thrust)) * aaT;

	// attitude part components
	FLOAT er;
	Eigen::Matrix<FLOAT, 1, 2> Der_upgoal;
	Eigen::Matrix<FLOAT, 1, 2> Der_up;
	std::tie(er, Der_upgoal, Der_up) = angleto(upgoal, up);

	FLOAT ew = w - w_d;
	FLOAT torque = -kr * er - kw * ew;
	Action u {thrust, torque};

	// controller chain rules
	auto Dthrust_x = Dthrust_a.transpose() * Da_x;
	auto Dthrust_th = Dthrust_a.transpose() * Da_th;

	//auto Der_x = Der_up * Dup_x + Der_upgoal * Dupgoal_a * Da_x;
	Eigen::Matrix<FLOAT, 1, XDIM> Der_x = Der_upgoal * Dupgoal_a * Da_x;
	Der_x(0, 6) += Der_up.dot(Dup_r);

	auto Der_th = Der_upgoal * Dupgoal_a * Da_th;

	Eigen::Matrix<FLOAT, 1, XDIM> Dtorque_xw;
	Dtorque_xw << 0, 0, 0, 0, 0, 0, 0, -kw;

	auto Dtorque_x = -kr * Der_x + Dtorque_xw;
	Eigen::Matrix<FLOAT, 1, TDIM> Dtorque_th;
	Dtorque_th << 0, 0, 0, -er, -ew;
	Dtorque_th += -kr * Der_th;

	Jux Du_x;
	Du_x << Dthrust_x, Dtorque_x;

	Jut Du_th;
	Du_th << Dthrust_th, Dtorque_th;

	return std::make_tuple(u, Du_x, Du_th);
}
*/

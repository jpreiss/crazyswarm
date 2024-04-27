#include <cmath>
#include <tuple>

#include <Eigen/Dense>


using FLOAT = double;
int constexpr XDIM = 8;
int constexpr UDIM = 2;
int constexpr TDIM = 5;
FLOAT constexpr GRAV = 9.81;

using Vec = Eigen::Matrix<FLOAT, 2, 1>;
using Mat = Eigen::Matrix<FLOAT, 2, 2, Eigen::RowMajor>;

using Jxx = Eigen::Matrix<FLOAT, XDIM, XDIM, Eigen::RowMajor>;
using Jxu = Eigen::Matrix<FLOAT, XDIM, UDIM, Eigen::RowMajor>;
using Jut = Eigen::Matrix<FLOAT, UDIM, TDIM, Eigen::RowMajor>;
using Jux = Eigen::Matrix<FLOAT, UDIM, XDIM, Eigen::RowMajor>;

using State = std::tuple<Vec, Vec, Vec, FLOAT, FLOAT>;
using Input = std::tuple<FLOAT, FLOAT>;

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

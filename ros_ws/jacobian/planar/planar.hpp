#include <cmath>
#include <tuple>

#include <Eigen/Dense>


using Vec = Eigen::Matrix<double, 2, 1>;
using Mat = Eigen::Matrix<double, 2, 2, Eigen::RowMajor>;

// Returns angle rotating vec. a to vec. b and gradients.
// Vectors must be unit length.
// Returns: angle, dangle/da, dangle/db
std::tuple<double, Vec, Vec> angleto(Vec a, Vec b)
{
	// Computes atan2 of b in the coordinate frame of a.
	double ax = a[0];
	double ay = a[1];
	double bx = b[0];
	double by = b[1];
	Mat R;
	R <<
		 ax, ay,
		-ay, ax;
	Vec Rb = R * b;
	Mat DRb_a;
	DRb_a <<
		bx,  by,
		by, -bx;
	double angle = std::atan2(Rb[1], Rb[0]);
	Vec grad_atan2 {-Rb[1], Rb[0]};
	Vec grad_a = grad_atan2.transpose() * DRb_a;
	Vec grad_b = grad_atan2.transpose() * R;

	return std::make_tuple(angle, grad_a, grad_b);
}

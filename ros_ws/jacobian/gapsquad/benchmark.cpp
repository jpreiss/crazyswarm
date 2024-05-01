#include <string>

#define EIGEN_INITIALIZE_MATRICES_BY_ZERO
using FLOAT = float;
#include "gapsquad.hpp"


int main(int argc, char const **argv)
{
	State x;
	Target t;
	Action u;
	Param th;
	FLOAT dt;

	if (argc == 1 || argv[1] == std::string("ctrl")) {
		for (int i = 0; i < 10000000; ++i) {
			Action u;
			Jux Du_x;
			Jut Du_th;
			ctrl(x, t, th, u, Du_x, Du_th);
		}
	}

	if (argc == 1 || argv[1] == std::string("dynamics")) {
		State xt;
		Jxx Dx_x;
		Jxu Dx_u;
		for (int i = 0; i < 10000000; ++i) {
			dynamics(x, t, u, dt, xt, Dx_x, Dx_u);
		}
	}
}

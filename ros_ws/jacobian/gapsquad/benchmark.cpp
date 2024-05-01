#include <string>

#include "gapsquad.hpp"


int main(int argc, char const **argv)
{
	Vec ierr; Vec p; Vec v; Mat R; Vec w; // state
	Vec p_d; Vec v_d; Vec a_d; FLOAT y_d; Vec w_d; // target
	FLOAT ki; FLOAT kp; FLOAT kv; FLOAT kr; FLOAT kw; // params

	if (argc == 1 || argv[1] == std::string("ctrl")) {
		for (int i = 0; i < 10000000; ++i) {
			ctrl(
				ierr, p, v, R, w, // state
				p_d, v_d, a_d, y_d, w_d, // target
				ki, kp, kv, kr, kw // params
			);
		}
	}

	FLOAT thrust; Vec torque; // action
	FLOAT dt; // constants

	if (argc == 1 || argv[1] == std::string("dynamics")) {
		for (int i = 0; i < 10000000; ++i) {
			dynamics(
				ierr, p, v, R, w, // state
				p_d, v_d, a_d, y_d, w_d, // target
				thrust, torque, // action
				dt // constants
			);
		}
	}
}

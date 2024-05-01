#include <string>

#include "gapsquad.hpp"


int main(int argc, char const **argv)
{
	Vec ierr; Vec p; Vec v; Mat R; Vec w; // state
	Vec p_d; Vec v_d; Vec a_d; FLOAT y_d; Vec w_d; // target
	FLOAT ki_xy; FLOAT ki_z; FLOAT kp_xy; FLOAT kp_z; FLOAT kv_xy; FLOAT kv_z; // position gains
	FLOAT kr_xy; FLOAT kr_z; FLOAT kw_xy; FLOAT kw_z; // attitude gains

	if (argc == 1 || argv[1] == std::string("ctrl")) {
		for (int i = 0; i < 10000000; ++i) {
			ctrl(
				ierr, p, v, R, w, // state
				p_d, v_d, a_d, y_d, w_d, // target
				ki_xy, ki_z, kp_xy, kp_z, kv_xy, kv_z, kr_xy, kr_z, kw_xy, kw_z // params
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

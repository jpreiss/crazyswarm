#include "gapsquad.hpp"

int main()
{
	Vec ierr; Vec p; Vec v; Mat R; Vec w; // state
	Vec p_d; Vec v_d; Vec a_d; FLOAT y_d; Vec w_d; // target
	FLOAT ki; FLOAT kp; FLOAT kv; FLOAT kr; FLOAT kw; // params

	for (int i = 0; i < 10000000; ++i) {
		ctrl(
			ierr, p, v, R, w, // state
			p_d, v_d, a_d, y_d, w_d, // target
			ki, kp, kv, kr, kw // params
		);
	}
}

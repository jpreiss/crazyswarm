#pragma once

#include <stdint.h>

// NOTE: FLOAT must resolve to a floating-point type. Add a `using` before
// including this header, i.e.
//
//     using FLOAT = double;
//     #include "gapsquad.hpp"
//
// We parameterize the float type because the finite differences used to check
// our analytic derivatives cannot be computed precisely enough in single
// precision. We need double for debugging and single for running on firmware.

// externally visible types / constants
#define XDIM (3 + 3 + 3 + 9 + 3)
#define UDIM (1 + 3)
// 5 params, each with xy and z variants
#define TDIM (2 * 5)

#ifdef __cplusplus
	#include <Eigen/Dense>
	using Vec = Eigen::Matrix<FLOAT, 3, 1>;
	using Mat = Eigen::Matrix<FLOAT, 3, 3, Eigen::RowMajor>;
#else
	#include "math3d.h"
	typedef struct vec Vec;
	typedef struct mat33 Mat;
	//typedef FLOAT[XDIM][XDIM] Jxx;
	//typedef FLOAT[XDIM][UDIM] Jxu;
	//typedef FLOAT[UDIM][TDIM] Jxt;
	//typedef FLOAT[UDIM][XDIM] Jux;
#endif

struct State { Vec ierr; Vec p; Vec v; Mat R; Vec w; };
struct Action { FLOAT thrust; Vec torque; };
struct Target { Vec p_d; Vec v_d; Vec a_d; FLOAT y_d; Vec w_d; };
struct Param {
	FLOAT ki_xy; FLOAT ki_z; FLOAT kp_xy; FLOAT kp_z; FLOAT kv_xy; FLOAT kv_z; // position gains
	FLOAT kr_xy; FLOAT kr_z; FLOAT kw_xy; FLOAT kw_z; // attitude gains
};
struct CostParam {
	FLOAT p; FLOAT v; FLOAT w; FLOAT thrust; FLOAT torque;
};

enum gaps_optimizer {
	GAPS_OPT_DISABLE = 0,
	GAPS_OPT_GRAD = 1,
	GAPS_OPT_ADADELTA = 2,
};

struct GAPS
{
	// main state
	Vec ierr;
	struct Param theta;
	FLOAT y[XDIM][TDIM];

	// main params
	struct CostParam cost_param;
	float eta;
	float damping;
	uint8_t optimizer;

	// diagnostics
	FLOAT yabsmax;

	// AdaDelta state
	FLOAT grad_accum[TDIM];
	FLOAT update_accum[TDIM];
	// AdaDelta params
	float ad_decay;
	float ad_eps;
};

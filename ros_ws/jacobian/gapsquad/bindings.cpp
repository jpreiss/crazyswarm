#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

using FLOAT = double;
#include "gapsquad.hpp"


using StateTuple = std::tuple<Vec, Vec, Vec, Mat, Vec>;
using ActionTuple = std::tuple<FLOAT, Vec>;
using TargetTuple = std::tuple<Vec, Vec, Vec, FLOAT, Vec>;
using ParamTuple = std::tuple<
	FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, FLOAT, // position gains
	FLOAT, FLOAT, FLOAT, FLOAT // attitude gains
>;
using CostParamTuple = std::tuple<FLOAT, FLOAT, FLOAT, FLOAT, FLOAT>;

// The C++ standard doesn't guarantee that this will hold, but it does in
// practice (at least on Clang++ 15), so we can use it to catch errors due to
// adding/removing a field.
static_assert(sizeof(StateTuple) == sizeof(State), "struct/tuple error");
static_assert(sizeof(ActionTuple) == sizeof(Action), "struct/tuple error");
static_assert(sizeof(TargetTuple) == sizeof(Target), "struct/tuple error");
static_assert(sizeof(ParamTuple) == sizeof(Param), "struct/tuple error");
static_assert(sizeof(CostParamTuple) == sizeof(CostParam), "struct/tuple error");
static_assert(sizeof(State) == sizeof(FLOAT) * XDIM, "size error");
static_assert(sizeof(Action) == sizeof(FLOAT) * UDIM, "size error");
static_assert(sizeof(Param) == sizeof(FLOAT) * TDIM, "size error");

std::tuple<ActionTuple, Jux, Jut>
ctrl_wrap(StateTuple const &xt, TargetTuple const &tt, ParamTuple const &tht)
{
	std::tuple<ActionTuple, Jux, Jut> output;

	State const &x = reinterpret_cast<State const &>(xt);
	Target const &t = reinterpret_cast<Target const &>(tt);
	Param const &th = reinterpret_cast<Param const &>(tht);
	Action &u = reinterpret_cast<Action &>(std::get<ActionTuple>(output));

	ctrl(x, t, th, u, std::get<Jux>(output), std::get<Jut>(output));

	return output;
}

std::tuple<StateTuple, Jxx, Jxu>
dynamics_wrap(StateTuple const &xt, TargetTuple const &tt, ActionTuple const &ut, FLOAT dt)
{
	std::tuple<StateTuple, Jxx, Jxu> output;

	State const &x = reinterpret_cast<State const &>(xt);
	Target const &t = reinterpret_cast<Target const &>(tt);
	Action const &u = reinterpret_cast<Action const &>(ut);
	State &xnext = reinterpret_cast<State &>(std::get<StateTuple>(output));

	dynamics(x, t, u, dt, xnext, std::get<Jxx>(output), std::get<Jxu>(output));

	return output;
}

std::tuple<FLOAT, Gcx, Gcu>
cost_wrap(StateTuple const &xt, TargetTuple const &tt, ActionTuple const &ut, CostParamTuple const &Qt) // inputs
{
	std::tuple<FLOAT, Gcx, Gcu> output;

	State const &x = reinterpret_cast<State const &>(xt);
	Target const &t = reinterpret_cast<Target const &>(tt);
	Action const &u = reinterpret_cast<Action const &>(ut);
	CostParam const &Q = reinterpret_cast<CostParam const &>(Qt);

	cost(x, t, u, Q, std::get<FLOAT>(output), std::get<Gcx>(output), std::get<Gcu>(output));

	return output;
}

namespace py = pybind11;

PYBIND11_MODULE(gapsquad, m) {
	// m.def("angleto", &angleto);
	m.def("ctrl", &ctrl_wrap);
	m.def("dynamics", &dynamics_wrap);
	m.def("cost", &cost_wrap);
	m.def("SO3error", &SO3error);
	m.def("cross", &cross);
	m.def("hat", &hat);
	m.def("normalize", &normalize);
}

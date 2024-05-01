#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

using FLOAT = double;
#include "gapsquad.hpp"

namespace py = pybind11;

PYBIND11_MODULE(gapsquad, m) {
    // m.def("angleto", &angleto);
    m.def("ctrl", &ctrl_wrap);
    m.def("dynamics", &dynamics_wrap);
    m.def("SO3error", &SO3error);
    m.def("cross", &cross);
    m.def("hat", &hat);
    m.def("normalize", &normalize);
}

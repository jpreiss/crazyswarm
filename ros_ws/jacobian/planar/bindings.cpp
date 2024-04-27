#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "planar.hpp"


PYBIND11_MODULE(planar, m) {
    m.def("angleto", &angleto);
}

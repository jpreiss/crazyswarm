jacobians.c: ctrl.py
	python3 ctrl.py > $@

ctrl.py: ctrl.c cmath3d.py c2py.py inputs.py computations.py
	echo "from cmath3d import *\n" > $@
	cat inputs.py >> $@
	cat ctrl.c | python3 c2py.py >> $@
	cat computations.py >> $@

EIGEN := $(shell python -c 'import eigenpip as e; print(e.get_include())')

benchmark: gapsquad/benchmark.cpp gapsquad/gapsquad.hpp
	$(CXX) $(CXXFLAGS) -g -O3 -I$(EIGEN) -I$(EIGEN)/unsupported --std=c++14 -DEIGEN_NO_MALLOC -fstack-usage -o $@ $<

.PHONY: stack

stack: benchmark
	@echo ctrl: `cat benchmark.su | grep ctrl | cut -f 2-`
	@echo dynamics: `cat benchmark.su | grep dynamics | cut -f 2-`

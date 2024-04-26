jacobians.c: ctrl.py
	python3 ctrl.py > $@

ctrl.py: ctrl.c cmath3d.py c2py.py inputs.py computations.py
	echo "from cmath3d import *\n" > $@
	cat inputs.py >> $@
	cat ctrl.c | python3 c2py.py >> $@
	cat computations.py >> $@

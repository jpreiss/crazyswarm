Installation
============

For real hardware operation, we assume that you have **Ubuntu 16.04 with ROS Kinetic** or **Ubuntu 18.04 with ROS Melodic** or **Ubuntu 20.04 with ROS Noetic** .
Avoid using a virtual machine because this adds additional latency and might cause issues with the visualization tools.

For simulation-only operation, **MacOS** is also supported.

.. warning::

    Using ubuntu in `Windows Subsystem for Linux (WSL) <https://docs.microsoft.com/en-us/windows/wsl/about>`_ is not supported since WSL does not have USB support and so Crazyradio will not work.
    You must install Ubuntu either directly on the computer or in a VM.


Simulation Only
---------------

It is possible to write/debug ``pycrazyswarm`` scripts and selected firmware modules
on a machine that does not have ROS or the ARM cross-compilation toolchain installed.
You can install just the components required for the simulation by doing the following:

----

Linux or MacOS with Anaconda
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not already have a version of ``conda`` installed on your system, we recommend the ``Miniconda`` distribution.
First, install `the Python 2.7 or 3.x version <https://docs.conda.io/en/latest/miniconda.html>`_.
Next, clone the Crazyswsarm repo and build the Anaconda environment with the specific python version number::

    $ git clone https://github.com/USC-ACTLab/crazyswarm.git
    $ cd crazyswarm
    $ conda create --name crazyswarm python=[DESIRED PYTHON VERSION]
    $ conda env update -f conda_env.yaml

Activate the Anaconda environment.
Then, set the ``CSW_PYTHON`` environment variable to either ``python`` or ``python3`` and run the build script::

    $ conda activate crazyswarm
    $ CSW_PYTHON=[DESIRED PYTHON COMMAND] ./buildSimOnly.sh

----

Linux without Anaconda
~~~~~~~~~~~~~~~~~~~~~~

Install the dependencies.
Set the ``CSW_PYTHON`` environment variable to either ``python`` or ``python3`` and clone the repository::

    $ export CSW_PYTHON=python3
    $ sudo apt install git make gcc swig lib${CSW_PYTHON}-dev ${CSW_PYTHON}-numpy ${CSW_PYTHON}-yaml ${CSW_PYTHON}-matplotlib
    $ git clone https://github.com/USC-ACTLab/crazyswarm.git
    $ cd crazyswarm
    $ ./buildSimOnly.sh

----

In either case, to test the installation, run one of the examples::

    $ cd ros_ws/src/crazyswarm/scripts
    $ python figure8_csv.py --sim

More details on the usage can be found in the :ref:`Usage` section.


Simulation and Physical Robots
------------------------------

For real hardware operation, we assume that you have one of the following three configurations, with the ROS "Desktop-Full" or "Desktop" configuration.

- Ubuntu 16.04, Python2, ROS Kinetic (`ROS setup instructions <http://wiki.ros.org/kinetic/Installation/Ubuntu>`_).
- Ubuntu 18.04, Python2, ROS Melodic (`ROS setup instructions <http://wiki.ros.org/melodic/Installation/Ubuntu>`_).
- Ubuntu 20.04, Python3, ROS Noetic (`ROS setup instructions <http://wiki.ros.org/noetic/Installation/Ubuntu>`_).

After setting up ROS, install the other dependencies.
Set the ``CSW_PYTHON`` environment variable to either ``python`` or ``python3`` and clone the repository::

    $ export CSW_PYTHON=python3
    $ sudo apt install git swig lib${CSW_PYTHON}-dev ${CSW_PYTHON}-numpy ${CSW_PYTHON}-yaml ${CSW_PYTHON}-matplotlib gcc-arm-embedded libpcl-dev libusb-1.0-0-dev sdcc
    $ git clone https://github.com/USC-ACTLab/crazyswarm.git
    $ cd crazyswarm

You can now build everything by running our build script.::

    $ ./build.sh

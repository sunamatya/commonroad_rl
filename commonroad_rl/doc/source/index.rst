.. CommonRoad_RL documentation master file, created by
   sphinx-quickstart on Tue Jul 10 09:17:31 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=============
CommonRoad-RL
=============

This project contains a software package to solve motion planning problems on CommonRoad
using reinforcement learning methods, currently based on `OpenAI Stable Baselines <https://stable-baselines.readthedocs.io/en/master/>`__.

The software is written in Python 3.6 and tested on Linux. The usage of the Anaconda_ Python distribution is strongly recommended.

.. _Anaconda: http://www.anaconda.com/download/#download


.. seealso::
	
	* `CommonRoad <https://commonroad.in.tum.de/>`__
	* `CommonRoad-io <https://commonroad.in.tum.de/commonroad_io>`__
	* `CommonRoad Drivability Checker <https://commonroad.in.tum.de/drivability_checker>`__
	* `Vehicle Models <https://gitlab.lrz.de/tum-cps/commonroad-vehicle-models/tree/master>`__
	* `Spinning Up <https://spinningup.openai.com/en/latest/>`__
	* `OpenAI Gym <https://gym.openai.com/docs/>`__
	* `OpenAI Safety Gym <https://openai.com/blog/safety-gym/>`__


.. Requirements
   ============

   The required dependencies for running CommonRoad_io are:

   * numpy>=1.13
   * shapely>=1.6.4
   * matplotlib>=2.2.2
   * lxml>=4.2.2
   * networkx>=2.2
   * Pillow>=7.0.0

Prerequisits
============

This project should be run with `conda <https://www.anaconda.com/>`__. Make sure it is installed before proceeding with the installation.
To create an environment for this project including all requirements, run::

	conda env create -n cr36 -f environment.yml

Installation
============

Currently, the package can only be installed from the repository. First, clone it::

	git clone https://gitlab.lrz.de/ss20-mpfav-rl/commonroad-rl.git

After the repository is cloned, CommonRoad_rl can be installed without sudo rights with::

	bash scripts/install.sh -e cr36 --no-root

and with sudo rights::

	bash scripts/install.sh -e cr36

:code:`cr36` to be replaced by the name of your conda environment if needed.

This will build all softwares in your home folder. You can press ``ctrl`` + ``c`` to skip when asked for sudo password.
Note that all necessary libraries need to be installed with sudo rights beforehands.
Please ask your admin to install them for you if they are missing.

Changelog
============

 

Getting Started
===============

A tutorial on the main functionalities can be found in the form of jupyter notebooks in the :code:`tutorials` folder.


.. toctree::
   :maxdepth: 6
   :caption: Contents:

   documentation/index.rst
   module/index.rst


.. Indices and tables
   ==================

   * :ref:`genindex`
   * :ref:`modindex`
   * :ref:`search`

   Contact information
   ===================

   :Website: `http://commonroad.in.tum.de <https://commonroad.in.tum.de/>`_
   :Email: `commonroad@lists.lrz.de <commonroad@lists.lrz.de>`_

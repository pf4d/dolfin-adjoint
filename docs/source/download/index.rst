.. _download:

**************************
Downloading dolfin-adjoint
**************************

Dependencies
============

Mandatory dependencies:
-----------------------

- `FEniCS`_, version 1.5. For installation instructions for FEniCS/dolfin, see `their installation instructions`_.

- `libadjoint`_. This is a library written in C that manipulates the tape of the forward model to derive the associated adjoint equations.

Optional dependencies:
----------------------

- `SLEPc`_. This is necessary if you want to conduct :doc:`generalised stability analyses <../documentation/gst>`.

- `IPOPT`_ and `pyipopt`_: This is the best available open-source optimisation algorithm. Strongly recommended if you wish to solve :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Make sure to compile IPOPT against the `Harwell Subroutine Library`_.

- `Moola`_: A set of optimisation algorithms specifically designed for :doc:`PDE-constrained optimisation problems <../documentation/optimisation>`. Install with `pip install moola`. Note: still experimental.

.. _FEniCS: http://fenicsproject.org
.. _libadjoint: http://bitbucket.org/dolfin-adjoint/libadjoint
.. _SLEPc: http://www.grycap.upv.es/slepc/
.. _IPOPT: https://projects.coin-or.org/Ipopt
.. _pyipopt: https://github.com/xuy/pyipopt
.. _moola: https://github.com/funsim/moola
.. _Harwell Subroutine Library: http://www.hsl.rl.ac.uk/ipopt/
.. _their installation instructions: http://fenicsproject.org/download

Virtual machine
===============

If you'd like to try dolfin-adjoint out without any installation headaches,
try out `our VirtualBox virtual machine with dolfin-adjoint 1.4 installed
<http://dolfin-adjoint.org/_static/dolfin-adjoint-1.4.ova>`_. Here are
the instructions:

* Download and install VirtualBox from https://www.virtualbox.org, or from your operating system.
* Download the `virtual machine <http://dolfin-adjoint.org/_static/dolfin-adjoint-1.4.ova>`_.
* Start VirtualBox, click on "File -> Import Appliance", select the virtual machine image and click on "Import".
* Select the "dolfin-adjoint VM" and click on "Start" to boot the machine.
* For installing new software you need the login credentials:

  * Username: fenics
  * Password: dolfinadjoint

Binary packages
===============

Binary packages are currently available for Ubuntu users through the
`launchpad PPA`_.  To install dolfin-adjoint, do

.. code-block:: bash

   sudo apt-add-repository ppa:libadjoint/ppa
   sudo apt-get update
   sudo apt-get install python-dolfin-adjoint

which should install the latest stable version on your system.
Once that's done, why not try out the :doc:`tutorial <../documentation/tutorial>`?

.. _launchpad PPA: https://launchpad.net/~libadjoint/+archive/ppa

From source
===========

The latest stable release of dolfin-adjoint and libadjoint is **version 1.5** which is compatible with FEniCS 1.5. Download links:

* libadjoint: `https://bitbucket.org/dolfin-adjoint/libadjoint/get/libadjoint-1.5.zip`_
* dolfin-adjoint: `https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/get/dolfin-adjoint-1.5.zip`_

.. _https://bitbucket.org/dolfin-adjoint/libadjoint/get/libadjoint-1.5.zip: https://bitbucket.org/dolfin-adjoint/libadjoint/get/libadjoint-1.5.zip
.. _https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/get/dolfin-adjoint-1.5.zip: https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/get/dolfin-adjoint-1.5.zip

The development version is available from `bitbucket`_ with the following
command:

.. code-block:: bash

   hg clone https://bitbucket.org/dolfin-adjoint/dolfin-adjoint#dolfin-adjoint-1.5

The development version of libadjoint is also available from bitbucket with the
following command:

.. code-block:: bash

   hg clone https://bitbucket.org/dolfin-adjoint/libadjoint#libadjoint-1.5

As dolfin-adjoint is a pure Python module, once its dependencies are
installed the development version can be used without system-wide
installation via

.. code-block:: bash

   export PYTHONPATH=<path to dolfin-adjoint>:$PYTHONPATH

Contributions (such as handling new features of dolfin, or new test
cases or examples) are very welcome.

.. _bitbucket: https://bitbucket.org/dolfin-adjoint/dolfin-adjoint

Older versions
==============

An older version, that is compatible with FEniCS 1.4 can be downloaded with:

* libadjoint: `https://bitbucket.org/dolfin-adjoint/libadjoint/downloads/libadjoint-1.4.tar.gz`_
* dolfin-adjoint: `https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/downloads/dolfin-adjoint-1.4.tar.gz`_

.. _https://bitbucket.org/dolfin-adjoint/libadjoint/downloads/libadjoint-1.4.tar.gz: https://bitbucket.org/dolfin-adjoint/libadjoint/downloads/libadjoint-1.4.tar.gz
.. _https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/downloads/dolfin-adjoint-1.4.tar.gz: https://bitbucket.org/dolfin-adjoint/dolfin-adjoint/downloads/dolfin-adjoint-1.4.tar.gz

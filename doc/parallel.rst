.. _parallel-operations:

===================
Parallel Operations
===================

OpenClimateGIS operations may be run in parallel using MPI provided ``mpi4py`` is installed. By default, data is distributed across a single spatial dimension. Dimensions are distributed (local bounds calculation) using the :class:`~ocgis.vmachine.mpi.OcgDist` class. Parallel execution is controlled by the :class:`~ocgis.OcgVM`.

All OpenClimateGIS operations are implemented in parallel with minimal inter-process communication. All `writes` are performed synchronously until suitable asynchronous IO packages are available for Python. OpenClimateGIS's parallelism handles extra ranks before and after subsetting through support for empty objects. Extra ranks following a subset are handled using subcommunicators by the :class:`~ocgis.OcgVM`.

For a standard OpenClimateGIS operations script, the script should be executed using ``mpirun``:

>>> mpirun -n <nprocs> </path/to/ocgis/script.py>

Parallelization Scheme
----------------------

OpenClimateGIS uses data parallelism for operations. Reading, subsetting, and calculations (the operations) are fully parallel. Multiple request datasets or subset geometries are processed in sequence for each dataset/geometry combination.

Spatial Averaging in Parallel
-----------------------------

OpenClimateGIS has no bit-for-bit guarantees when spatial averaging in parallel. Each rank spatially averages its own data before combining rank sums on the root process. This often leads to floating point summation differences from a spatial average performed in serial. Hence, if the floating point errors affect your analysis, it is recommended that the process be run in serial or an advanced regridding application like `ESMF <http://www.earthsystemmodeling.org/esmf_releases/last_built/ESMF_refdoc/node5.html#SECTION05011000000000000000>`_ is used.

.. _parallel-example:

Explanatory Parallel Example
----------------------------

This examples executes a simple subset using OpenClimateGIS data interface objects and the :class:`~ocgis.OcgVM` in parallel. The code should be executed using ``mpirun -n 2 </path/to/script.py>``.

.. literalinclude:: sphinx_examples/explanatory_parallel_example.py

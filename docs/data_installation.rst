Data Installation
=================

``nSTAT-python`` does not bundle raw example data in the Git tree.
The canonical paper-example dataset is downloaded automatically the first
time a paper example or dataset helper requires it.

Use one of the supported Python-native prefetch paths if you want the cache
materialized ahead of time:

Command line
------------

.. code-block:: bash

   nstat-install --download-example-data always

Python API
----------

.. code-block:: python

   from nstat.data_manager import ensure_example_data

   data_dir = ensure_example_data(download=True)
   print(data_dir)

Notes
-----

- The dataset source is figshare DOI ``10.6084/m9.figshare.4834640.v3``.
- Source checkouts cache data under ``data_cache/nstat_data`` by default.
- Set ``NSTAT_DATA_DIR`` to point at an existing dataset cache if needed.
- The repository intentionally ignores ``data/`` and ``data_cache/`` so local downloads are not committed.

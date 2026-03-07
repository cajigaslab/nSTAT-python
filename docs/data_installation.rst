Data Installation
=================

``nSTAT-python`` does not bundle raw example data in the Git tree.

Use one of the supported Python-native installation paths instead:

Command line
------------

.. code-block:: bash

   nstat-install --download-example-data always

Module invocation also works:

.. code-block:: bash

   python -m nstat.install --download-example-data never --no-rebuild-doc-search

Python API
----------

.. code-block:: python

   from nstat.data_manager import ensure_example_data

   data_dir = ensure_example_data(download=True)
   print(data_dir)

MATLAB-compatible paper-data helper
-----------------------------------

.. code-block:: python

   from nstat import getPaperDataDirs, get_paper_data_dirs

   data_dir, mepsc_dir, explicit_stimulus_dir, psth_dir, place_cell_data_dir = getPaperDataDirs()
   dirs = get_paper_data_dirs()
   assert dirs.data_dir == data_dir

Notes
-----

- Example data is cached under ``data_cache/`` by default.
- Set ``NSTAT_DATA_DIR`` to point at an existing dataset cache if needed.
- The repository intentionally ignores ``data/`` so local example-data installs are not committed.
- ``clean_user_path_prefs`` is accepted only as a MATLAB-compatibility no-op.

{{ fullname | escape | underline }}

{#
   ``class_targets`` (a plain dict built in docs/conf.py and passed via
   ``autosummary_context``) maps the public name to the class's real home.  Several public names
   are aliases (``nstat.CovColl = CovariateCollection``).  autodoc renders
   an alias as just "alias of ..." with no members, which is what registers
   the public name (so the api.rst table entry links).  For aliases we then
   also render the real class's members on the same page, marked no-index so
   the class is not registered twice (duplicate object description).

   No nested "Methods"/"Attributes" ``.. autosummary::`` tables (the stock
   template emits them): those resolve ``CovColl.add`` against same-named
   shim modules (``nstat.CovColl`` is a module AND the class) and fail.
   ``autodoc_default_options`` already renders every member.

   ``:inherited-members:`` on the non-alias ``autoclass``: thin subclasses
   (``nstColl``, ``Covariate``, ``FitResSummary``, ...) define almost nothing
   themselves, so without it their pages list no members at all.
#}
{% set t = class_targets[fullname] %}
.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{%- if not t.is_alias %}
   :inherited-members:

   .. automethod:: __init__
{%- else %}

.. autoclass:: {{ t.module }}.{{ t.qualname }}
   :no-index:

   .. automethod:: __init__
      :no-index:
{%- endif %}

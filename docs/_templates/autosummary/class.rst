{{ fullname | escape | underline }}

{#
   ``class_target`` (defined in docs/conf.py via ``autosummary_context``)
   resolves the public name to the class's real home.  Several public names
   are aliases (``nstat.CovColl = CovariateCollection``).  autodoc renders
   an alias as just "alias of ..." with no members, which is what registers
   the public name (so the api.rst table entry links).  For aliases we then
   also render the real class's members on the same page, marked no-index so
   the class is not registered twice (duplicate object description).

   No nested "Methods"/"Attributes" ``.. autosummary::`` tables (the stock
   template emits them): those resolve ``CovColl.add`` against same-named
   shim modules (``nstat.CovColl`` is a module AND the class) and fail.
   ``autodoc_default_options`` already renders every member.
#}
{% set t = class_target(fullname) %}
.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
{%- if not t.is_alias %}

   .. automethod:: __init__
{%- else %}

.. autoclass:: {{ t.module }}.{{ t.qualname }}
   :no-index:

   .. automethod:: __init__
      :no-index:
{%- endif %}

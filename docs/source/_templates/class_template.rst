{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members: Module, MoveDataMixin, LinearOperator

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

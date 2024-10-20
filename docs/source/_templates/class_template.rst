{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members: Module
   :show-inheritance:

   {% block methods %}
   .. automethod:: __init__
   {% endblock %}

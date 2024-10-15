from mrpro import algorithms, operators, data, phantoms, utils

def _populate_all_recursive():
    """
    Recursively populates __all__ for all modules and submodules in the given package,
    including only members defined within each module.
    """
    import sys
    import pkgutil
    import types

    package = sys.modules[__name__]

    def _populate_all_for_module(module_name):
        module = sys.modules[module_name]
        module.__all__ = [name for name, obj in module.__dict__.items()
                          if (isinstance(obj, (types.FunctionType, type)) and obj.__module__ == module_name)
                          or (not name.startswith('_'))]

    for _, modname, _ in pkgutil.walk_packages(package.__path__, __name__ + '.'):
        __import__(modname)
        _populate_all_for_module(modname)

    _populate_all_for_module(__name__)
    del sys.modules[__name__]._populate_all_recursive

_populate_all_recursive()
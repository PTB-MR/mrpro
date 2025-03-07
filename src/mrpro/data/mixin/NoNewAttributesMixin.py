"""Mixin to prevent creation of new attributes."""

from typing import Any


class NoNewAttributesMixin:
    """A mixin that prevents the creation of new attributes after initialization."""

    def __init_subclass__(cls, *args, no_new_attributes: bool = True, **kwargs):
        """Apply to subclass."""
        super().__init_subclass__(**kwargs)
        if not no_new_attributes:
            return

        original_post_init = vars(cls).get('__post_init__')

        def new_post_init(self: NoNewAttributesMixin) -> None:
            if original_post_init:
                original_post_init(self)
            self.__initialized = True

        cls.__post_init__ = new_post_init  # type: ignore[attr-defined]

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        """Set an attribute."""
        if not hasattr(self, name) and hasattr(self, '_NoNewAttributesMixin__initialized'):
            raise AttributeError(f'Cannot set attribute {name} on {self.__class__.__name__}')
        super().__setattr__(name, value)

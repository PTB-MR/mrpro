"""Building blocks for numerical phantoms."""

from dataclasses import dataclass


@dataclass(slots=True)
class EllipseParameters:
    """Parameters of ellipse."""

    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    intensity: float

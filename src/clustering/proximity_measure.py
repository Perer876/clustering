import numpy as np
from clustering.common import Vector


def euclidean(x: Vector, y: Vector) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(np.array(x) - np.array(y)))

from clustering.common import Vector, Cluster
import numpy as np


def mean(cluster: Cluster) -> Vector:
    """Mean representative of a cluster."""
    return np.mean(cluster, axis=0)

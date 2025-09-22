from typing import final, Sequence
import numpy as np
from clustering.common import Cluster, Distance, Representative
from clustering.proximity_measure import euclidean
from clustering.representative import mean


@final
class Bsas:
    def __init__(
        self,
        q: int,
        th: float,
        d: Distance = euclidean,
        r: Representative = mean,
    ):
        """
        Basic Sequential Algorithmic Scheme (BSAS).
        :param q: maximum number of clusters.
        :param th: threshold distance for creating a new cluster.
        :param d: distance function.
        :param r: representative function.
        """
        self.q = q
        self.th = th
        self.d = d
        self.r = r

    def __call__(self, data: Cluster) -> Sequence[int]:
        pass

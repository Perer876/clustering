from typing import final, Sequence
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

    def __call__(self, x: Cluster) -> Sequence[int]:
        n = len(x)

        if n == 0:
            return []

        m = 1
        clusters = [[x[0]]]
        representatives = [x[0]]
        labels = [0]

        for i in range(1, n):
            xi = x[i]

            d_clusters = [
                (k, ck, self.d(xi, representatives[k])) for k, ck in enumerate(clusters)
            ]

            k, ck, distance = min(d_clusters, key=lambda d_cluster: d_cluster[0])

            if distance > self.th and m < self.q:
                labels.append(m)
                m += 1
                clusters.append([xi])
                representatives.append(xi)
            else:
                labels.append(k)
                ck.append(xi)
                representatives[k] = self.r(ck)

        return labels

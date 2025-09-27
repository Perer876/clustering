from typing import final, Final, Sequence
from clustering.common import Cluster
from clustering.proximity_measure import euclidean
import numpy as np


@final
class KMeans:
    def __init__(self, k: int, seed: int = None):
        """
        K-Means clustering algorithm.
        :param k: number of clusters.
        """
        self.k: Final = k
        self.rng = np.random.default_rng(seed)

    def __call__(self, x: Cluster) -> Sequence[int]:
        x = np.array(x)

        n = len(x)

        if n < self.k:
            raise ValueError("The number of samples must be at least equal to k.")

        c = self.rng.choice(x, self.k)
        u = np.zeros((self.k, n), dtype=int)

        converged = False
        while not converged:
            # First stage
            for i in range(n):
                xi = x[i]

                distances = [euclidean(xi, cj) for cj in c]
                closest = np.argmin(distances)

                u[closest][i] = 1

            # Second stage
            converged = True
            for i in range(self.k):
                ci = c[i]
                ui = u[i]
                updated_ci = np.sum(ui.reshape(n, 1) * x, axis=0) / np.sum(ui)

                if not np.array_equal(ci, updated_ci):
                    converged = False
                    c[i] = updated_ci

        return u.argmax(axis=0).tolist()

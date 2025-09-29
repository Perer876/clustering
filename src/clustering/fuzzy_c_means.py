from typing import final, Final, Sequence
from clustering.common import Cluster
import numpy as np


@final
class FuzzyCMeans:
    def __init__(
        self,
        c: int,
        m: float,
        max_iter: int = 100,
        seed: int = None,
    ):
        """
        Fuzzy C-Means clustering algorithm.
        :param c: number of clusters.
        :param m: fuzziness parameter (m > 1).
        :param max_iter: maximum number of iterations.
        :param seed: random seed for centroid initialization.
        """
        self.c: Final = c
        self.m: Final = m
        self.max_iter: Final = max_iter
        self.rng: Final = np.random.default_rng(seed)

    def __call__(self, x: Cluster) -> Sequence[Sequence[float]]:
        x = np.array(x)

        n = len(x)
        two_over_m_minus_one = 2 / (self.m - 1)

        if n < self.c:
            raise ValueError(
                f"The number of samples ({n}) must be at least equal to c ({self.c})"
            )

        v = self.rng.choice(x, self.c)
        w = np.zeros((self.c, n), dtype=float)

        converged = False
        iteration = 0
        while not converged and iteration < self.max_iter:
            # First stage
            for i in range(n):
                xi = x[i]

                l2_norm_xi_minus_v = np.linalg.norm(xi - v, axis=1)

                # Handle the case where xi is exactly equal to a centroid
                j = np.argmin(l2_norm_xi_minus_v)
                if l2_norm_xi_minus_v[j] == 0:
                    w[:, i] = 0
                    w[j, i] = 1
                    continue

                for j in range(self.c):
                    w[j, i] = 1 / np.sum(
                        (l2_norm_xi_minus_v[j] / l2_norm_xi_minus_v)
                        ** two_over_m_minus_one
                    )

            # Second stage
            converged = True
            for j in range(self.c):
                vj = v[j]
                wj = w[j]

                updated_vj = np.sum((wj**self.m).reshape(n, 1) * x, axis=0) / np.sum(
                    wj**self.m
                )

                if not np.array_equal(vj, updated_vj):
                    converged = False
                    v[j] = updated_vj

            iteration += 1

        # Membership matrix transposed to have shape (n, c) instead of (c, n)
        return w.T

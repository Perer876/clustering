from typing import final, Callable

type Vector = tuple[int, ...]
type Cluster = tuple[Vector, ...]
type Distance = Callable[[Vector, Cluster], float]

@final
class Bsas:
    def __init__(
        self,
        q: int,
        th: float,
        d: Distance,
    ):
        self.q = q
        self.th = th
        self.d = d

    def __call__(self, data: tuple[Vector, ...]) -> tuple[Cluster, ...]:
        pass


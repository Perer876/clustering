from typing import Callable, Sequence

type Vector = Sequence[float]
type Cluster = Sequence[Vector]
type Distance = Callable[[Vector, Vector], float]
type Representative = Callable[[Cluster], Vector]

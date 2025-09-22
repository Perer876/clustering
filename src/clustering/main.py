from typer import Typer
from rich.console import Console
from pathlib import Path
import numpy as np

app = Typer()
console = Console()


@app.command()
def make_blobs(
    file: Path,
    samples: int = 100,
    features: int = 2,
    centers: int = 3,
    seed: int = None,
):
    """Generate a cluster."""
    from sklearn.datasets import make_blobs

    blobs, _ = make_blobs(
        n_samples=samples,
        n_features=features,
        centers=centers,
        random_state=seed,
    )

    np.savetxt(file, blobs, delimiter=",")


@app.command()
def bsas(
    input: Path,
    output: Path,
    q: int,
    th: float,
):
    """Basic Sequential Algorithmic Scheme (BSAS) clustering."""
    pass

from typer import Typer
from rich.console import Console
from pathlib import Path
import numpy as np

app = Typer()
console = Console()


@app.command()
def create_dataset(
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
    threshold: float = 1.0,
    max_clusters: int = 4,
):
    """
    Basic Sequential Algorithmic Scheme (BSAS) clustering.
    :param threshold: threshold distance for creating a new cluster.
    :param max_clusters: maximum number of clusters.
    """
    from clustering.bsas import Bsas

    _bsas = Bsas(q=max_clusters, th=threshold)

    cluster = np.loadtxt(input, delimiter=",")

    labels = _bsas(cluster)

    labeled_cluster = np.column_stack((np.array(labels), cluster))

    if output.is_dir():
        file_path = output / input.name
    else:
        file_path = output

    np.savetxt(file_path, labeled_cluster, delimiter=",")

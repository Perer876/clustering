from typer import Typer
from rich.console import Console
from pathlib import Path
import numpy as np

from clustering.utils import file_destination

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
def plot(from_file: Path, to_file: Path):
    """Plot a 2D cluster."""
    import matplotlib.pyplot as plt

    cluster = np.loadtxt(from_file, delimiter=",")

    if cluster.shape[1] != 2:
        console.print("[red]Error:[/red] The input file must have 2 columns (x, y).")
        return

    plt.scatter(cluster[:, 0], cluster[:, 1], c="gray", marker="o")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cluster Plot")
    plt.grid(True)

    plt.savefig(to_file)


@app.command()
def plot_clusters(from_file: Path, to_file: Path):
    """Plot 2D clusters."""
    import matplotlib.pyplot as plt

    cluster = np.loadtxt(from_file, delimiter=",")

    if cluster.shape[1] != 3:
        console.print(
            "[red]Error:[/red] The input file must have 3 columns (label, x, y)."
        )
        return

    labels = cluster[:, 0]
    points = cluster[:, 1:]

    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Cluster Plot")
    plt.grid(True)

    plt.savefig(to_file)


@app.command()
def bsas(
    origin: Path,
    target: Path,
    threshold: float = 1.0,
    max_clusters: int = 4,
):
    """
    Basic Sequential Algorithmic Scheme (BSAS) clustering.
    :param origin: path to the input file.
    :param target: path to the output file or directory.
    :param threshold: threshold distance for creating a new cluster.
    :param max_clusters: maximum number of clusters.
    """
    from clustering.bsas import Bsas

    _bsas = Bsas(q=max_clusters, th=threshold)

    cluster = np.loadtxt(origin, delimiter=",")

    labels = _bsas(cluster)

    labeled_cluster = np.column_stack((np.array(labels), cluster))

    file_path = file_destination(origin, target)

    np.savetxt(file_path, labeled_cluster, delimiter=",")


@app.command()
def k_means(
    origin: Path,
    target: Path,
    k: int = 4,
    seed: int = None,
):
    """
    K-Means clustering.
    :param origin: path to the input file.
    :param target: path to the output file or directory.
    :param k: number of clusters.
    :param seed: random seed for centroid initialization.
    """
    from clustering.k_means import KMeans

    _k_means = KMeans(k=k, seed=seed)

    cluster = np.loadtxt(origin, delimiter=",")

    labels = _k_means(cluster)

    labeled_cluster = np.column_stack((np.array(labels), cluster))

    file_path = file_destination(origin, target)

    np.savetxt(file_path, labeled_cluster, delimiter=",")

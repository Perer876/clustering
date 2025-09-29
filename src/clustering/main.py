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
def plot_fuzzy_clusters(from_file: Path, to_file: Path):
    """Plot 2D fuzzy clusters."""
    import matplotlib.pyplot as plt

    cluster = np.loadtxt(from_file, delimiter=",")

    if cluster.shape[1] < 3:
        console.print(
            "[red]Error:[/red] The input file must have at least 3 columns (membership weights..., x, y)."
        )
        return

    points = cluster[:, -2:]
    membership_weights = cluster[:, :-2]
    labels = np.argmax(membership_weights, axis=1)

    plt.scatter(points[:, 0], points[:, 1], c=labels, cmap="tab10")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Fuzzy Cluster Plot")
    plt.grid(True)

    n, m = membership_weights.shape

    base_colors = plt.cm.tab10(
        np.linspace(0, 1, m)
    )  # Cambia a plt.cm.hsv o similar si m>10

    # Computar colores mixtos
    colors = np.zeros((n, 4))  # RGBA
    for j in range(m):
        colors += membership_weights[:, j][:, np.newaxis] * base_colors[j]

    # Clip para seguridad
    colors = np.clip(colors, 0, 1)

    plt.figure(figsize=(8, 6))
    plt.scatter(points[:, 0], points[:, 1], c=colors, alpha=0.7)

    # Agregar leyenda con colores base
    for j in range(m):
        plt.plot(
            [],
            [],
            color=base_colors[j],
            marker="o",
            linestyle="None",
            label=f"Cluster {j + 1}",
        )

    plt.title("Agrupamiento Fuzzy K-Means (colores mixtos por pertenencia)")
    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")

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


@app.command()
def fuzzy_c_means(
    origin: Path,
    target: Path,
    c: int = 4,
    m: float = 2.0,
    seed: int = None,
):
    """
    Fuzzy C-Means clustering.
    :param origin: path to the input file.
    :param target: path to the output file or directory.
    :param c: number of clusters.
    :param m: fuzziness parameter (m > 1).
    :param seed: random seed for centroid initialization.
    """
    from clustering.fuzzy_c_means import FuzzyCMeans

    _fuzzy_c_means = FuzzyCMeans(c=c, m=m, seed=seed)

    cluster = np.loadtxt(origin, delimiter=",")

    membership_weights = _fuzzy_c_means(cluster)

    labeled_cluster = np.column_stack((membership_weights, cluster))

    file_path = file_destination(origin, target)

    np.savetxt(file_path, labeled_cluster, delimiter=",")

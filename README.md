# Clustering

This repository holds some clustering utilities used for a subject in a Master's degree in Applied Computing.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) tool to manage python virtual environment and dependencies.

## Setup

1. Clone the repository:

    ```bash
    git clone git@github.com:Perer876/clustering.git
    cd clustering
    ```

2. Create environment, install dependencies and install cli:

    ```bash
    uv tool install ./ --editable
    ```

## Usage

```bash
clustering --help
```

Create blobs dataset:

```bash
clustering create-dataset data/raw/blobs.csv --samples 100 --features 2 --centers 3 --seed 1
```

Cluster using bsas:

```bash
clustering bsas data/raw/blobs.csv data/processed/blobs_bsas.csv --threshold 1.5 --max-clusters 5
```

Plot a dataset:

```bash
clustering plot data/raw/blobs.csv plots/blobs.png
```

### K-means

```bash
clustering k-means data/raw/100S_2F_3C.csv data/processed/k-means_3k.csv --k 3 --seed 0
clustering plot-clusters data/processed/k-means_3k.csv plots/k-means_3k.png
```
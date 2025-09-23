from pathlib import Path


def file_destination(origin: Path, target: Path) -> Path:
    if origin.is_dir():
        return target / origin.name
    else:
        return target

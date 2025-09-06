import pathlib


def prepare_sample_paths(base_dir: pathlib.Path | str, stem: str):
    base = pathlib.Path(base_dir)
    raw = str(base / f"{stem}.mp4")
    track = base / f"{stem}.json"
    return raw, track

from pathlib import Path
from typing import Tuple, Union


def prepare_sample_paths(data_dir: Union[str, Path], filename: str) -> Tuple[str, Path]:
    base = Path(data_dir)
    raw_candidate = (base / f"{filename}.mp4").resolve()
    tracking_path = (base / f"{filename}.json").resolve()
    return str(raw_candidate), tracking_path



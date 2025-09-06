from pathlib import Path
from typing import Tuple, Union


def prepare_sample_paths(data_dir: Union[str, Path], filename: str) -> Tuple[str, Path]:
    base = Path(data_dir)
    raw_candidate = (base / f"{filename}.mp4").resolve()
    tracking_path = (base / f"{filename}.json").resolve()
    return str(raw_candidate), tracking_path



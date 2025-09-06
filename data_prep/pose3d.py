from typing import List
import numpy as np
import torch


def infer_clips(
    model_3d: torch.nn.Module,
    clips: List[np.ndarray],
    device,
    flip_fn,
    idx_maps: List[np.ndarray] | None = None,
) -> List[np.ndarray]:
    """
    Run 3D model over clips with flip-ensemble and root-centering.

    Args:
        model_3d: torch model
        clips: list of (1,243,17,3) float32 arrays
        device: torch device
        flip_fn: function that flips the input tensor across x and swaps LR joints
        idx_maps: list of index maps to restore original frame counts, or None

    Returns:
        list of (T,17,3) numpy arrays per clip
    """
    outputs_3d: List[np.ndarray] = []
    model_3d.eval()
    with torch.no_grad():
        for ci, clip in enumerate(clips):
            t_in = torch.from_numpy(clip).to(device)

            out_nf = model_3d(t_in)
            out_f = model_3d(flip_fn(t_in))
            out = 0.5 * (out_nf + flip_fn(out_f))

            # root-center by subtraction
            out = out - out[:, :, 0:1, :]

            if idx_maps is not None and idx_maps[ci] is not None:
                out = out[:, idx_maps[ci]]

            outputs_3d.append(out[0].detach().cpu().numpy())
    return outputs_3d



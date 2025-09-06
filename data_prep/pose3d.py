from typing import List
import numpy as np
import torch
import torch.nn as nn
from .temporal import to_overlapping_clips
from .constants import CLIP_LEN


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


def combine_overlapping_predictions(
    preds: List[np.ndarray],
    index_maps: List[np.ndarray],
) -> np.ndarray:
    """
    Average overlapping per-frame predictions into a single (F,17,3) array.
    """
    if len(preds) == 0:
        return np.zeros((0, 17, 3), dtype=np.float32)
    max_index = int(max(int(m.max()) for m in index_maps))
    F = max_index + 1
    accum = np.zeros((F, 17, 3), dtype=np.float64)
    # shape (F,1,1) to broadcast with (F,17,3)
    counts = np.zeros((F, 1, 1), dtype=np.float64)
    for clip_pred, idx_map in zip(preds, index_maps):
        for t_local, t_global in enumerate(idx_map):
            accum[t_global] += clip_pred[t_local]
            counts[t_global, 0, 0] += 1.0
    counts[counts == 0.0] = 1.0
    combined = (accum / counts).astype(np.float32)
    return combined


def infer_3d_with_overlap(
    model_3d: torch.nn.Module,
    inp_xyc: np.ndarray,
    device,
    flip_fn,
    target: int = 243,
    hop: int | None = None,
) -> np.ndarray:
    """
    Full 3D pipeline: create overlapping clips from (1,F,17,3) input, run 3D inference with flip-ensemble,
    and average overlapping predictions into a single (F,17,3) array.
    """
    clips, index_maps = to_overlapping_clips(inp_xyc, target=target, hop=hop)
    preds = infer_clips(model_3d=model_3d, clips=clips, device=device, flip_fn=flip_fn, idx_maps=None)
    combined = combine_overlapping_predictions(preds, [m for m in index_maps])
    return combined


def build_motionagformer(MotionAGFormer, device, ckpt_path) -> torch.nn.Module:
    n_layers, dim_in, dim_feat, dim_rep, dim_out = 16, 3, 128, 512, 3
    mlp_ratio, act_layer = 4, nn.GELU
    attn_drop, drop, drop_path = 0.0, 0.0, 0.0
    use_layer_scale, layer_scale_init_value, use_adaptive_fusion = True, 1e-5, True
    num_heads, qkv_bias, qkv_scale = 8, False, None
    hierarchical = False
    use_temporal_similarity, neighbour_num, temporal_connection_len = True, 2, 1
    use_tcn, graph_only = False, False
    n_frames = CLIP_LEN

    model_3d = nn.DataParallel(
        MotionAGFormer(
            n_layers=n_layers,
            dim_in=dim_in,
            dim_feat=dim_feat,
            dim_rep=dim_rep,
            dim_out=dim_out,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            attn_drop=attn_drop,
            drop=drop,
            drop_path=drop_path,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            use_adaptive_fusion=use_adaptive_fusion,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_scale=qkv_scale,
            hierarchical=hierarchical,
            use_temporal_similarity=use_temporal_similarity,
            neighbour_num=neighbour_num,
            temporal_connection_len=temporal_connection_len,
            use_tcn=use_tcn,
            graph_only=graph_only,
            n_frames=n_frames,
        )
    ).to(device)

    pre_dict = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    assert 'model' in pre_dict, "Checkpoint missing 'model' key"
    model_3d.load_state_dict(pre_dict['model'], strict=True)
    model_3d.eval()
    return model_3d



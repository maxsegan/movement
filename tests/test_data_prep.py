import sys
import types
import numpy as np
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))


def test_prepare_sample_paths(tmp_path):
    from data_prep.paths import prepare_sample_paths
    raw, track = prepare_sample_paths(tmp_path, "clip")
    assert raw == str(tmp_path / "clip.mp4")
    assert track == tmp_path / "clip.json"

def test_resample():
    from data_prep.temporal import resample
    assert resample(10, target=5).tolist() == [0, 2, 4, 6, 8]
    assert resample(3, target=5).tolist() == [0, 0, 1, 1, 2]

def test_to_overlapping_clips():
    from data_prep.temporal import to_overlapping_clips
    arr = np.zeros((1, 5, 17, 3), dtype=np.float32)
    clips, idx_maps = to_overlapping_clips(arr, target=3, hop=2)
    assert len(clips) == 3
    assert [idx.tolist() for idx in idx_maps] == [[0, 1, 2], [2, 3, 4], [4, 4, 4]]
    assert all(c.shape == (1, 3, 17, 3) for c in clips)

def test_normalize_screen():
    from data_prep.geometry import normalize_screen
    xy = np.array([[0, 0], [640, 480], [320, 240]], dtype=np.float32)
    out = normalize_screen(xy, 640, 480)
    exp = np.array([[-1.0, -0.75], [1.0, 0.75], [0.0, 0.0]], dtype=np.float32)
    assert np.allclose(out, exp)

def test_fit_similarity_2d():
    from data_prep.geometry import fit_similarity_2d
    src = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    R = np.array([[0, -1], [1, 0]], dtype=np.float32)
    scale = 2.0
    t = np.array([3, 4], dtype=np.float32)
    dst = (src @ R) * scale + t
    s, R_est, t_est = fit_similarity_2d(src, dst)
    assert np.isclose(s, scale)
    assert np.allclose(R_est, R)
    assert np.allclose(t_est, t)

def test_normalize_to_bbox():
    from data_prep.geometry import normalize_to_bbox
    kpts = np.zeros((1, 1, 17, 2), dtype=np.float32)
    kpts[0, 0, 0] = [10, 15]
    boxes = np.array([[0, 0, 20, 30]], dtype=np.float32)
    out = normalize_to_bbox(kpts, boxes)
    assert np.allclose(out[0, 0, 0], [0.0, 0.0])

def test_build_input_xyc():
    from data_prep.geometry import build_input_xyc
    seq_kpts = np.zeros((1, 2, 17, 2), dtype=np.float32)
    seq_kpts[0, 0, 0] = [0, 0]
    seq_kpts[0, 1, 0] = [640, 480]
    seq_scores = np.full((1, 2, 17), 0.5, dtype=np.float32)
    inp = build_input_xyc(seq_kpts, seq_scores, 640, 480)
    assert inp.shape == (1, 2, 17, 3)
    assert np.allclose(inp[0, 0, 0], [-1.0, -0.75, 0.5])
    assert np.allclose(inp[0, 1, 0], [1.0, 0.75, 0.5])

def test_coco_h36m():
    from data_prep.keypoints import coco_h36m
    keypoints = np.stack(
        [np.stack([[j, j + 100] for j in range(17)], axis=0)], axis=0
    ).astype(np.float32)
    h36m, valid = coco_h36m(keypoints)
    assert h36m.shape == (1, 17, 2)
    assert valid.tolist() == [0]
    assert np.allclose(h36m[0, 11], [5.0, 105.0])

def test_h36m_coco_format():
    from data_prep.keypoints import h36m_coco_format
    base = np.stack([[j, j + 100] for j in range(17)], axis=0).astype(np.float32)
    kpts = base[None, None, :, :]
    scores = np.ones((1, 1, 17), dtype=np.float32)
    out_kpts, out_scores, valid = h36m_coco_format(kpts, scores)
    assert out_kpts.shape == (1, 1, 17, 2)
    assert out_scores.shape == (1, 1, 17)
    assert np.array_equal(valid[0], np.array([0]))
    assert np.allclose(out_scores, 1.0)

def test_combine_overlapping_predictions():
    torch_stub = types.ModuleType("torch")
    nn_stub = types.ModuleType("torch.nn")
    class Module:
        pass
    nn_stub.Module = Module
    torch_stub.nn = nn_stub
    sys.modules.setdefault("torch", torch_stub)
    sys.modules.setdefault("torch.nn", nn_stub)

    from data_prep.pose3d import combine_overlapping_predictions

    pred1 = np.ones((3, 17, 3), dtype=np.float32)
    pred2 = np.full((3, 17, 3), 2.0, dtype=np.float32)
    idx_maps = [np.array([0, 1, 2]), np.array([1, 2, 3])]
    combined = combine_overlapping_predictions([pred1, pred2], idx_maps)
    assert combined.shape == (4, 17, 3)
    assert np.allclose(combined[0], 1.0)
    assert np.allclose(combined[1], 1.5)
    assert np.allclose(combined[2], 1.5)
    assert np.allclose(combined[3], 2.0)


def test_to_clips_resample_and_idx_map():
    from data_prep.temporal import to_clips
    N, F, J = 1, 5, 17
    arr = np.zeros((N, F, J, 3), dtype=np.float32)
    boxes = np.stack([np.array([0, 0, 10, 20], dtype=np.float32)] * F)
    clips, box_clips, idx_maps = to_clips(arr, boxes, target=8)
    assert len(clips) == 1 and clips[0].shape[1] == 8
    assert idx_maps[0] is not None
    down = np.unique(np.floor(np.linspace(0, F - 1, 8)).astype(np.int32), return_index=True)[1]
    assert np.array_equal(idx_maps[0], down)


def test_constants_clip_hop_default():
    from data_prep.constants import CLIP_LEN, CLIP_HOP
    assert CLIP_HOP == CLIP_LEN // 2

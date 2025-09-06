import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display
import cv2
from .constants import H36M_I, H36M_J, H36M_LR_EDGE_MASK as LR, COLOR_L_BGR as COLOR_L, COLOR_R_BGR as COLOR_R


def visualize_layered_heatmaps(heatmaps: np.ndarray):
    # Expect (B, J, H, W) or (J, H, W)
    if heatmaps.ndim == 4:
        heatmaps = heatmaps[0]
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected 3D array (J,H,W), got {heatmaps.shape}")
    combined_heatmap = np.sum(heatmaps, axis=0)
    plt.figure(figsize=(6, 5))
    plt.imshow(combined_heatmap, cmap='hot', interpolation='nearest', alpha=0.9)
    plt.colorbar()
    plt.title('Layered Heatmap Visualization')
    plt.axis('off')
    plt.show()


def plot_pose_equal_axes(pts: np.ndarray, elev: int = 15, azim: int = 70, title: str = "3D Pose (equal axes)"):
    # pts: (17,3) in model coordinate frame
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    I = np.array([0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15])
    J = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    c = pts[0]
    P = pts - c

    xyz_min = P.min(axis=0)
    xyz_max = P.max(axis=0)
    r = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    for a, b in zip(I, J):
        ax.plot([P[a, 0], P[b, 0]],
                [P[a, 1], P[b, 1]],
                [P[a, 2], P[b, 2]],
                lw=2, c='b')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=12, c='r')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def overlay_3d_on_video_inline(
    all_3d: list,
    seq_keypoints: np.ndarray,
    idxs: np.ndarray,
    raw_video_path: str,
    fit_similarity_2d_fn,
    stride: int = 2,
    resize_scale: float = 0.5,
    max_fps: float = 20.0,
):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import animation

    assert len(all_3d) > 0 and isinstance(all_3d, (list, tuple)), "all_3d must be a non-empty list"
    y3d = np.concatenate(all_3d, axis=0)  # (T3, 17, 3)
    T = min(y3d.shape[0], len(idxs), seq_keypoints.shape[1])
    if T < len(idxs):
        print(f"Note: 3D predictions length ({y3d.shape[0]}) < tracked frames ({len(idxs)}); truncating to {T}.")

    H36M_I = np.array([0,1,2,0,4,5,0,7,8,9,8,11,12,8,14,15])
    H36M_J = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    cap0 = cv2.VideoCapture(raw_video_path)
    fps_src = cap0.get(cv2.CAP_PROP_FPS) or 30.0
    cap0.release()

    frames_rgb = []
    for s in range(0, T, int(max(1, stride))):
        frame_no = int(idxs[s])
        cap = cv2.VideoCapture(raw_video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            continue

        P = y3d[s]
        X = P[:, :2].astype(np.float32)
        Z = P[:, 2].astype(np.float32)
        Y = seq_keypoints[0, s].astype(np.float32)

        scale_xy, R, t = fit_similarity_2d_fn(X, Y)
        mapped = (X @ R) * scale_xy + t  # (17,2) pixels

        z_edge = ((Z[H36M_I] + Z[H36M_J]) * 0.5)
        order = np.argsort(z_edge)

        canvas = frame_bgr.copy()
        for idx_e in order:
            i = int(H36M_I[idx_e]); j = int(H36M_J[idx_e])
            p1 = tuple(np.round(mapped[i]).astype(int))
            p2 = tuple(np.round(mapped[j]).astype(int))
            cv2.line(canvas, p1, p2, (0, 180, 255), 2, cv2.LINE_AA)
        for k in range(mapped.shape[0]):
            p = tuple(np.round(mapped[k]).astype(int))
            cv2.circle(canvas, p, 3, (255, 255, 0), -1, cv2.LINE_AA)

        if resize_scale != 1.0:
            canvas = cv2.resize(
                canvas,
                (int(canvas.shape[1] * resize_scale), int(canvas.shape[0] * resize_scale)),
                interpolation=cv2.INTER_AREA,
            )

        frames_rgb.append(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))

    fps_out = float(min(max_fps, fps_src / max(1.0, float(stride))))
    fig, ax = plt.subplots(figsize=(frames_rgb[0].shape[1]/160, frames_rgb[0].shape[0]/160), dpi=160)
    ax.axis('off')
    im = ax.imshow(frames_rgb[0])

    def update(i):
        im.set_data(frames_rgb[i])
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(frames_rgb), interval=1000.0/float(fps_out), blit=True)
    html = anim.to_html5_video()
    display(HTML(html))
    plt.close(fig)
    print(f"Rendered inline 3D overlay (no files). frames={len(frames_rgb)} stride={stride} scale={resize_scale} fps~{fps_out:.1f}")


def overlay_pose_and_bbox(rgb_frame: np.ndarray, xy: np.ndarray, scores: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    import numpy as np
    import supervision as sv

    annot_img = rgb_frame.copy()
    keypoints = sv.KeyPoints(xy=xy.astype(np.float32), confidence=scores.astype(np.float32))
    detections = sv.Detections(xyxy=np.expand_dims(np.array(xyxy, dtype=np.float32), axis=0))

    edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=3)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=2)
    bounding_box_annotator = sv.BoxAnnotator(color=sv.Color.BLUE, color_lookup=sv.ColorLookup.INDEX, thickness=1)

    annot_img = edge_annotator.annotate(scene=annot_img, key_points=keypoints)
    annot_img = vertex_annotator.annotate(scene=annot_img, key_points=keypoints)
    annot_img = bounding_box_annotator.annotate(scene=annot_img, detections=detections)
    return annot_img


def draw_h36m_on_video_frame(
    raw_video_path: str,
    idxs: np.ndarray,
    seq_keypoints: np.ndarray,
    seq_scores: np.ndarray,
    seq_idx: int = 0,
    conf_thresh: float = 0.15,
    radius: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    cap = cv2.VideoCapture(raw_video_path)
    frame_no = int(idxs[seq_idx])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Failed to read frame {frame_no}")

    pts = seq_keypoints[0, seq_idx].copy()
    scores = seq_scores[0, seq_idx].copy()
    h, w = frame_bgr.shape[:2]
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

    canvas = frame_bgr.copy()
    for i, j, is_left in zip(H36M_I, H36M_J, LR):
        if scores[i] < conf_thresh or scores[j] < conf_thresh:
            continue
        p1 = tuple(np.round(pts[i]).astype(int))
        p2 = tuple(np.round(pts[j]).astype(int))
        color = COLOR_L if is_left else COLOR_R
        cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

    for k in range(pts.shape[0]):
        if scores[k] < conf_thresh:
            continue
        p = tuple(np.round(pts[k]).astype(int))
        cv2.circle(canvas, p, radius, (0, 255, 255), -1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

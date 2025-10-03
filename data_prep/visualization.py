"""
Unified visualization module combining basic and comparison visualization functions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
from IPython.display import HTML, display
import logging

from .constants import H36M_I, H36M_J, H36M_LR_EDGE_MASK as LR, COLOR_L_BGR as COLOR_L, COLOR_R_BGR as COLOR_R

logger = logging.getLogger(__name__)

# Joint names for H36M format
H36M_JOINT_NAMES = [
    'Hip', 'R_Hip', 'R_Knee', 'R_Ankle', 'L_Hip', 'L_Knee', 'L_Ankle',
    'Spine', 'Neck', 'Head', 'HeadTop',
    'L_Shoulder', 'L_Elbow', 'L_Wrist', 'R_Shoulder', 'R_Elbow', 'R_Wrist'
]


# ============================================================================
# Basic Visualization Functions (from visualization.py)
# ============================================================================

def visualize_layered_heatmaps(heatmaps: np.ndarray):
    """Visualize combined heatmaps from multiple joints."""
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


def plot_pose_3d(
    pts: np.ndarray,
    elev: int = 15,
    azim: int = 70,
    title: str = "3D Pose",
    ax: Optional[plt.Axes] = None,
    color: str = 'b',
    marker_color: str = 'r'
) -> Optional[plt.Axes]:
    """
    Plot a single 3D pose with equal axes.

    Args:
        pts: (17, 3) joint positions
        elev: Elevation angle for 3D view
        azim: Azimuth angle for 3D view
        title: Plot title
        ax: Optional existing axis to plot on
        color: Color for bones
        marker_color: Color for joint markers

    Returns:
        The axis object if ax was None
    """
    from mpl_toolkits.mplot3d import Axes3D

    # Center pose at hip
    c = pts[0]
    P = pts - c

    # Calculate axis limits
    xyz_min = P.min(axis=0)
    xyz_max = P.max(axis=0)
    r = float(np.max(xyz_max - xyz_min) / 2.0 + 1e-6)

    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        return_ax = True
    else:
        return_ax = False

    # Plot skeleton
    for a, b in zip(H36M_I, H36M_J):
        ax.plot([P[a, 0], P[b, 0]],
                [P[a, 1], P[b, 1]],
                [P[a, 2], P[b, 2]],
                lw=2, c=color)

    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=12, c=marker_color)

    # Set equal aspect ratio
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([-r, r])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    return ax if return_ax else None


def overlay_pose_and_bbox(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: Optional[np.ndarray] = None,
    bbox: Optional[np.ndarray] = None,
    score_threshold: float = 0.15,
    joint_size: int = 2,
    bone_width: int = 2,
    bbox_color: Tuple[int, int, int] = (255, 0, 0)
) -> np.ndarray:
    """
    Overlay 2D pose and bounding box on a frame.

    Args:
        frame: Input image (H, W, 3)
        keypoints: 2D joint positions (17, 2)
        scores: Joint confidence scores (17,)
        bbox: Bounding box [x1, y1, x2, y2]
        score_threshold: Minimum score to show joint
        joint_size: Size of joint markers
        bone_width: Width of bone connections
        bbox_color: Color for bounding box

    Returns:
        Frame with overlay
    """
    img = frame.copy()

    # Draw bounding box
    if bbox is not None and np.all(np.isfinite(bbox)):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), bbox_color, 2)

    # Draw skeleton
    if keypoints is not None:
        for i, j in zip(H36M_I, H36M_J):
            if scores is None or (scores[i] >= score_threshold and scores[j] >= score_threshold):
                p1 = (int(keypoints[i, 0]), int(keypoints[i, 1]))
                p2 = (int(keypoints[j, 0]), int(keypoints[j, 1]))

                # Color based on left/right
                color = COLOR_L if LR[i] == 0 else COLOR_R
                cv2.line(img, p1, p2, color, bone_width, cv2.LINE_AA)

        # Draw joints
        for j in range(keypoints.shape[0]):
            if scores is None or scores[j] >= score_threshold:
                p = (int(keypoints[j, 0]), int(keypoints[j, 1]))
                cv2.circle(img, p, joint_size, (0, 255, 255), -1, cv2.LINE_AA)

    return img


# ============================================================================
# Comparison Visualization Functions (from comparison_visualization.py)
# ============================================================================

def create_comparison_colormap():
    """Create a colormap for similarity visualization (red=bad, yellow=medium, green=good)."""
    colors = ['red', 'yellow', 'green']
    return LinearSegmentedColormap.from_list('similarity', colors)


def plot_pose_comparison_3d(
    predicted_pose: np.ndarray,
    ground_truth_pose: np.ndarray,
    similarity_scores: Optional[np.ndarray] = None,
    title: str = "Pose Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot 3D comparison between predicted and ground truth poses side by side.

    Args:
        predicted_pose: (17, 3) predicted pose
        ground_truth_pose: (17, 3) ground truth pose
        similarity_scores: Optional (17,) per-joint similarity scores
        title: Plot title
        save_path: Optional path to save the plot
    """
    fig = plt.figure(figsize=(15, 5))

    # Plot predicted
    ax1 = fig.add_subplot(131, projection='3d')
    plot_pose_3d(predicted_pose, ax=ax1, title="Predicted", color='b', marker_color='r')

    # Plot ground truth
    ax2 = fig.add_subplot(132, projection='3d')
    plot_pose_3d(ground_truth_pose, ax=ax2, title="Ground Truth", color='g', marker_color='darkgreen')

    # Plot overlay if similarity scores provided
    if similarity_scores is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        plot_pose_with_similarity(predicted_pose, similarity_scores, ax=ax3, title="Similarity")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_pose_with_similarity(
    pose: np.ndarray,
    similarity_scores: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "Pose with Similarity"
) -> None:
    """
    Plot 3D pose colored by similarity scores.

    Args:
        pose: (17, 3) pose positions
        similarity_scores: (17,) similarity scores [0, 1]
        ax: Optional existing axis
        title: Plot title
    """
    from mpl_toolkits.mplot3d import Axes3D

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Center pose
    P = pose - pose[0:1]

    # Setup colormap
    cmap = create_comparison_colormap()

    # Plot bones colored by average joint similarity
    for i, j in zip(H36M_I, H36M_J):
        avg_score = (similarity_scores[i] + similarity_scores[j]) / 2
        color = cmap(avg_score)
        ax.plot([P[i, 0], P[j, 0]],
                [P[i, 1], P[j, 1]],
                [P[i, 2], P[j, 2]],
                lw=3, c=color)

    # Plot joints colored by similarity
    for j in range(pose.shape[0]):
        ax.scatter(P[j, 0], P[j, 1], P[j, 2],
                  s=50, c=[cmap(similarity_scores[j])],
                  edgecolors='black', linewidths=1)

    # Set limits
    r = np.max(np.abs(P)) * 1.1
    ax.set_xlim([-r, r])
    ax.set_ylim([-r, r])
    ax.set_zlim([-r, r])
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')


def plot_temporal_similarity(
    similarity_scores: np.ndarray,
    title: str = "Temporal Similarity",
    save_path: Optional[str] = None
) -> None:
    """
    Plot similarity scores over time.

    Args:
        similarity_scores: (T,) or (T, J) temporal similarity scores
        title: Plot title
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(12, 4))

    if similarity_scores.ndim == 1:
        # Single similarity score per frame
        ax.plot(similarity_scores, 'b-', linewidth=2)
        ax.fill_between(range(len(similarity_scores)),
                        similarity_scores, 0, alpha=0.3)
        ax.set_ylabel('Similarity Score')
    else:
        # Per-joint similarity scores
        im = ax.imshow(similarity_scores.T, aspect='auto',
                      cmap='RdYlGn', vmin=0, vmax=1)
        ax.set_ylabel('Joint Index')
        plt.colorbar(im, ax=ax, label='Similarity')

    ax.set_xlabel('Frame')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def plot_joint_similarity_heatmap(
    similarity_matrix: np.ndarray,
    joint_names: Optional[List[str]] = None,
    title: str = "Joint Similarity Heatmap",
    save_path: Optional[str] = None
) -> None:
    """
    Plot a heatmap of per-joint similarities.

    Args:
        similarity_matrix: (J, J) or (J,) similarity scores
        joint_names: Optional joint names
        title: Plot title
        save_path: Optional save path
    """
    if joint_names is None:
        joint_names = H36M_JOINT_NAMES

    fig, ax = plt.subplots(figsize=(10, 8))

    if similarity_matrix.ndim == 1:
        # Convert to 2D for display
        similarity_matrix = similarity_matrix.reshape(-1, 1)

    im = ax.imshow(similarity_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Set ticks
    ax.set_xticks(range(similarity_matrix.shape[1]))
    ax.set_yticks(range(similarity_matrix.shape[0]))

    if similarity_matrix.shape[1] == 1:
        ax.set_xticklabels(['Similarity'])
    else:
        ax.set_xticklabels(joint_names[:similarity_matrix.shape[1]], rotation=45, ha='right')

    ax.set_yticklabels(joint_names[:similarity_matrix.shape[0]])

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Similarity Score')

    # Add values in cells
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if similarity_matrix[i, j] < 0.5 else 'black')

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()


def create_comparison_summary_plot(
    predicted_sequence: np.ndarray,
    ground_truth_sequence: np.ndarray,
    similarity_result: Dict,
    save_path: Optional[str] = None,
    sample_frames: Optional[List[int]] = None
) -> None:
    """
    Create a comprehensive comparison summary plot.

    Args:
        predicted_sequence: (T, 17, 2) or (T, 17, 3) predicted poses
        ground_truth_sequence: (T, 17, 2) or (T, 17, 3) ground truth poses
        similarity_result: Dictionary with similarity metrics
        save_path: Optional save path
        sample_frames: Optional list of frame indices to visualize
    """
    fig = plt.figure(figsize=(16, 10))

    # Select sample frames
    if sample_frames is None:
        n_samples = min(4, predicted_sequence.shape[0])
        sample_frames = np.linspace(0, predicted_sequence.shape[0]-1, n_samples, dtype=int)

    # Plot sample pose comparisons
    n_cols = len(sample_frames)
    for i, frame_idx in enumerate(sample_frames):
        ax = fig.add_subplot(3, n_cols, i+1, projection='3d' if predicted_sequence.shape[-1] == 3 else None)

        if predicted_sequence.shape[-1] == 3:
            plot_pose_3d(predicted_sequence[frame_idx], ax=ax,
                        title=f"Frame {frame_idx}", color='b', marker_color='r')
        else:
            # 2D plot
            ax.scatter(predicted_sequence[frame_idx, :, 0],
                      predicted_sequence[frame_idx, :, 1], c='b', s=20)
            ax.set_title(f"Predicted Frame {frame_idx}")
            ax.set_aspect('equal')
            ax.invert_yaxis()

    # Plot temporal similarity
    ax_temporal = fig.add_subplot(3, 1, 2)
    if 'frame_similarities' in similarity_result:
        ax_temporal.plot(similarity_result['frame_similarities'], 'b-', linewidth=2)
        ax_temporal.fill_between(range(len(similarity_result['frame_similarities'])),
                                 similarity_result['frame_similarities'], 0, alpha=0.3)
        ax_temporal.set_xlabel('Frame')
        ax_temporal.set_ylabel('Similarity')
        ax_temporal.set_title('Temporal Similarity')
        ax_temporal.grid(True, alpha=0.3)

    # Plot per-joint similarity
    ax_joints = fig.add_subplot(3, 1, 3)
    if 'joint_similarities' in similarity_result:
        joint_sims = similarity_result['joint_similarities']
        bars = ax_joints.bar(range(len(joint_sims)), joint_sims)

        # Color bars by similarity
        cmap = create_comparison_colormap()
        for i, (bar, sim) in enumerate(zip(bars, joint_sims)):
            bar.set_color(cmap(sim))

        ax_joints.set_xlabel('Joint Index')
        ax_joints.set_ylabel('Average Similarity')
        ax_joints.set_title('Per-Joint Similarity')
        ax_joints.set_ylim([0, 1])
        ax_joints.grid(True, alpha=0.3, axis='y')

    # Add overall metrics
    metrics_text = f"Overall Similarity: {similarity_result.get('overall_similarity', 0):.3f}\n"
    if 'temporal_consistency' in similarity_result:
        metrics_text += f"Temporal Consistency: {similarity_result['temporal_consistency']:.3f}\n"
    if 'anomalous_frames' in similarity_result:
        metrics_text += f"Anomalous Frames: {len(similarity_result['anomalous_frames'])}"

    fig.text(0.02, 0.98, metrics_text, transform=fig.transFigure,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle("Pose Comparison Summary")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved summary plot to {save_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# Video Generation Functions
# ============================================================================

def create_comparison_video(
    frames: np.ndarray,
    predicted_poses: np.ndarray,
    ground_truth_poses: Optional[np.ndarray] = None,
    output_path: str = "comparison.mp4",
    fps: float = 30.0,
    scores: Optional[np.ndarray] = None
) -> None:
    """
    Create a video comparing predicted poses with ground truth.

    Args:
        frames: (T, H, W, 3) video frames
        predicted_poses: (T, 17, 2) predicted 2D poses
        ground_truth_poses: Optional (T, 17, 2) ground truth poses
        output_path: Output video path
        fps: Frames per second
        scores: Optional (T, 17) confidence scores
    """
    height, width = frames.shape[1:3]

    if ground_truth_poses is not None:
        # Side by side comparison
        out_width = width * 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_width, height))

        for t in range(frames.shape[0]):
            # Predicted overlay
            pred_frame = overlay_pose_and_bbox(
                frames[t],
                predicted_poses[t],
                scores[t] if scores is not None else None
            )

            # Ground truth overlay
            gt_frame = overlay_pose_and_bbox(
                frames[t],
                ground_truth_poses[t],
                bbox_color=(0, 255, 0)
            )

            # Concatenate side by side
            combined = np.concatenate([pred_frame, gt_frame], axis=1)
            writer.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    else:
        # Single video with overlay
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for t in range(frames.shape[0]):
            frame = overlay_pose_and_bbox(
                frames[t],
                predicted_poses[t],
                scores[t] if scores is not None else None
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    logger.info(f"Saved comparison video to {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def save_pose_plots_grid(
    poses: np.ndarray,
    titles: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    max_cols: int = 4
) -> None:
    """
    Save multiple poses in a grid layout.

    Args:
        poses: (N, 17, 3) multiple poses
        titles: Optional titles for each pose
        save_path: Save path
        max_cols: Maximum columns in grid
    """
    n_poses = poses.shape[0]
    n_cols = min(n_poses, max_cols)
    n_rows = (n_poses + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    for i in range(n_poses):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection='3d')
        title = titles[i] if titles else f"Pose {i}"
        plot_pose_3d(poses[i], ax=ax, title=title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()

    plt.close()
#!/usr/bin/env python3
"""
ByteTrack: Simple, Fast and Strong Multi-Object Tracker
Simplified implementation for person tracking in videos
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class Track:
    """Represents a single track (person) with Kalman filter for motion prediction"""

    _count = 0

    def __init__(self, bbox: np.ndarray, score: float):
        Track._count += 1
        self.track_id = Track._count
        self.bbox = bbox.copy()  # [x1, y1, x2, y2]
        self.score = score

        # Kalman filter state: [cx, cy, w, h, vx, vy]
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        self.mean = np.array([cx, cy, w, h, 0, 0], dtype=np.float32)
        self.covariance = np.eye(6, dtype=np.float32) * 10

        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = 'tracked'  # tracked, lost, removed

        # Color for visualization (consistent per track)
        np.random.seed(self.track_id)
        self.color = tuple(np.random.randint(50, 255, 3).tolist())

    def predict(self):
        """Predict next position using constant velocity model"""
        # State transition matrix
        F = np.array([
            [1, 0, 0, 0, 1, 0],  # cx = cx + vx
            [0, 1, 0, 0, 0, 1],  # cy = cy + vy
            [0, 0, 1, 0, 0, 0],  # w = w
            [0, 0, 0, 1, 0, 0],  # h = h
            [0, 0, 0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1],  # vy = vy
        ], dtype=np.float32)

        # Process noise
        Q = np.eye(6, dtype=np.float32)
        Q[4, 4] = 0.01  # Small noise for velocity
        Q[5, 5] = 0.01

        self.mean = F @ self.mean
        self.covariance = F @ self.covariance @ F.T + Q
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: np.ndarray, score: float):
        """Update track with new detection"""
        # Measurement
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # Calculate velocity
        vx = cx - self.mean[0]
        vy = cy - self.mean[1]

        # Measurement matrix
        z = np.array([cx, cy, w, h], dtype=np.float32)
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ], dtype=np.float32)

        # Measurement noise
        R = np.eye(4, dtype=np.float32) * 10

        # Kalman gain
        S = H @ self.covariance @ H.T + R
        K = self.covariance @ H.T @ np.linalg.inv(S)

        # Update
        y = z - H @ self.mean  # Innovation (4-dimensional)
        self.mean = self.mean + K @ y  # K is 6x4, y is 4x1 -> result is 6x1
        self.covariance = (np.eye(6) - K @ H) @ self.covariance

        # Update velocity estimate
        self.mean[4] = vx * 0.5 + self.mean[4] * 0.5  # Smooth velocity
        self.mean[5] = vy * 0.5 + self.mean[5] * 0.5

        self.bbox = bbox.copy()
        self.score = score
        self.hits += 1
        self.time_since_update = 0
        self.state = 'tracked'

    def get_predicted_bbox(self) -> np.ndarray:
        """Get predicted bounding box"""
        cx, cy, w, h = self.mean[:4]
        return np.array([
            cx - w/2,
            cy - h/2,
            cx + w/2,
            cy + h/2
        ], dtype=np.float32)


class ByteTracker:
    """
    ByteTrack: Association by confidence scores
    High confidence detections are matched first, then low confidence ones
    """

    def __init__(
        self,
        track_thresh: float = 0.5,    # High confidence threshold
        track_buffer: int = 30,       # Frames to keep lost tracks
        match_thresh: float = 0.8,    # IoU matching threshold
        min_box_area: float = 10,     # Minimum box area
    ):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.removed_tracks: List[Track] = []

        self.frame_id = 0
        Track._count = 0  # Reset track IDs

    def update(self, detections: np.ndarray, scores: np.ndarray) -> List[Track]:
        """
        Update tracks with new detections

        Args:
            detections: Nx4 array of [x1, y1, x2, y2]
            scores: N array of confidence scores

        Returns:
            List of active tracks
        """
        self.frame_id += 1

        # Filter by area
        if len(detections) > 0:
            areas = (detections[:, 2] - detections[:, 0]) * (detections[:, 3] - detections[:, 1])
            valid = areas > self.min_box_area
            detections = detections[valid]
            scores = scores[valid]

        # Separate high and low confidence detections
        if len(detections) > 0:
            high_idx = scores >= self.track_thresh
            low_idx = ~high_idx

            high_dets = detections[high_idx]
            high_scores = scores[high_idx]
            low_dets = detections[low_idx]
            low_scores = scores[low_idx]
        else:
            high_dets = np.empty((0, 4))
            high_scores = np.empty(0)
            low_dets = np.empty((0, 4))
            low_scores = np.empty(0)

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Separate tracked and lost tracks
        tracked = [t for t in self.tracks if t.state == 'tracked']
        lost = [t for t in self.tracks if t.state == 'lost']

        # Step 1: Match high confidence detections with tracked tracks
        if len(high_dets) > 0 and len(tracked) > 0:
            dists = self._iou_distance(
                np.array([t.get_predicted_bbox() for t in tracked]),
                high_dets
            )
            matches, u_tracks, u_dets = self._linear_assignment(dists, self.match_thresh)

            for m in matches:
                tracked[m[0]].update(high_dets[m[1]], high_scores[m[1]])

            # Unmatched tracks become lost
            for i in u_tracks:
                tracked[i].state = 'lost'
                lost.append(tracked[i])

            # Get unmatched high confidence detections
            high_dets = high_dets[u_dets]
            high_scores = high_scores[u_dets]
        else:
            # No matches, all tracks become lost
            for t in tracked:
                t.state = 'lost'
                lost.append(t)

        # Step 2: Match remaining high confidence with lost tracks
        if len(high_dets) > 0 and len(lost) > 0:
            dists = self._iou_distance(
                np.array([t.get_predicted_bbox() for t in lost]),
                high_dets
            )
            matches, u_tracks, u_dets = self._linear_assignment(dists, self.match_thresh)

            for m in matches:
                lost[m[0]].update(high_dets[m[1]], high_scores[m[1]])
                lost[m[0]].state = 'tracked'

            # Get unmatched detections
            high_dets = high_dets[u_dets]
            high_scores = high_scores[u_dets]

        # Step 3: Match low confidence with remaining tracks (tracked + lost)
        remaining_tracks = [t for t in self.tracks if t.time_since_update == 1]
        if len(low_dets) > 0 and len(remaining_tracks) > 0:
            dists = self._iou_distance(
                np.array([t.get_predicted_bbox() for t in remaining_tracks]),
                low_dets
            )
            matches, _, _ = self._linear_assignment(dists, 0.5)  # Lower threshold

            for m in matches:
                remaining_tracks[m[0]].update(low_dets[m[1]], low_scores[m[1]])
                remaining_tracks[m[0]].state = 'tracked'

        # Step 4: Create new tracks from unmatched high confidence detections
        for i in range(len(high_dets)):
            track = Track(high_dets[i], high_scores[i])
            self.tracks.append(track)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks
                      if t.time_since_update < self.track_buffer]

        # Return active tracks
        return [t for t in self.tracks if t.state == 'tracked']

    def _iou_distance(self, tracks_bbox: np.ndarray, dets_bbox: np.ndarray) -> np.ndarray:
        """
        Compute IoU distance matrix (1 - IoU)
        """
        if len(tracks_bbox) == 0 or len(dets_bbox) == 0:
            return np.empty((len(tracks_bbox), len(dets_bbox)))

        # Compute IoU
        ious = np.zeros((len(tracks_bbox), len(dets_bbox)))
        for i, track_box in enumerate(tracks_bbox):
            for j, det_box in enumerate(dets_bbox):
                ious[i, j] = self._iou(track_box, det_box)

        return 1 - ious

    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _linear_assignment(
        self,
        cost_matrix: np.ndarray,
        thresh: float
    ) -> Tuple[List, List, List]:
        """
        Linear assignment with threshold
        """
        if cost_matrix.size == 0:
            return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

        matches = []
        unmatched_tracks = []
        unmatched_dets = []

        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 1 - thresh:  # Convert back from distance to IoU
                matches.append([i, j])
            else:
                unmatched_tracks.append(i)
                unmatched_dets.append(j)

        # Find completely unmatched
        for i in range(cost_matrix.shape[0]):
            if i not in row_ind:
                unmatched_tracks.append(i)

        for j in range(cost_matrix.shape[1]):
            if j not in col_ind:
                unmatched_dets.append(j)

        return matches, unmatched_tracks, unmatched_dets

    def get_main_track(self) -> Optional[Track]:
        """
        Get the main track (most consistent, central, largest)
        """
        active_tracks = [t for t in self.tracks if t.state == 'tracked']
        if not active_tracks:
            return None

        # Score tracks by: hits (consistency) + centrality + size
        scores = []
        for track in active_tracks:
            # Consistency score
            consistency = track.hits / max(track.age, 1)

            # Centrality (assuming normalized coordinates)
            cx = (track.bbox[0] + track.bbox[2]) / 2
            cy = (track.bbox[1] + track.bbox[3]) / 2
            centrality = 1.0 - abs(cx - 0.5) - abs(cy - 0.5)

            # Size
            w = track.bbox[2] - track.bbox[0]
            h = track.bbox[3] - track.bbox[1]
            size = w * h

            score = consistency * 2 + centrality + size * 0.0001
            scores.append(score)

        return active_tracks[np.argmax(scores)]
# posture_tracker.py
import numpy as np
from collections import deque
from enum import Enum, auto

class PostureState(Enum):
    UPRIGHT = auto()
    LEANING_FORWARD = auto()
    UNKNOWN = auto()

    def __str__(self) -> str:
        """Return a nice human-readable name."""
        return self.name.replace("_", " ").title()
        # e.g., UPRIGHT -> "Upright", LEANING_FORWARD -> "Leaning Forward"

class PostureTracker:
    """
    Tracks posture by recording reference averages of keypoints
    and comparing them to live keypoint data.
    """

    def __init__(self, buffer_size: int = 30):
        """
        Args:
            buffer_size (int): Number of frames to keep for averaging live poses.
        """
        self.buffer_size = buffer_size
        self.live_buffer = deque(maxlen=buffer_size)
        self.references: dict[PostureState, np.ndarray] = {}
    
    def update_live_pose(self, keypoints: np.ndarray):
        """Update tracker with new live keypoints."""
        if not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints)
        self.live_buffer.append(keypoints)

    def get_live_average(self) -> np.ndarray | None:
        """Compute the average pose from the live buffer."""
        if len(self.live_buffer) == 0:
            return None
        return np.mean(np.stack(self.live_buffer), axis=0)

    def record_reference(self, state: PostureState = PostureState.UPRIGHT):
        """Save the current live average as a reference posture."""
        avg_pose = self.get_live_average()
        if avg_pose is not None:
            self.references[state] = avg_pose
            print(f"[PostureTracker] Recorded reference for {state.name}.")
        else:
            print("[PostureTracker] No live data to record.")

    def compare_to_reference(self, state: PostureState) -> float | None:
        """Compare live average pose to a stored reference posture."""
        live_avg = self.get_live_average()
        ref_pose = self.references.get(state, None)
        if live_avg is None or ref_pose is None:
            return None
        distances = np.linalg.norm(live_avg - ref_pose, axis=1)
        return float(np.mean(distances))

    def classify_posture(self, upright_ref: PostureState = PostureState.UPRIGHT, threshold: float = 30.0) -> PostureState:
        """Determine posture state relative to upright reference."""
        diff = self.compare_to_reference(upright_ref)
        if diff is None:
            return PostureState.UNKNOWN
        if diff < threshold:
            return PostureState.UPRIGHT
        else:
            return PostureState.LEANING_FORWARD
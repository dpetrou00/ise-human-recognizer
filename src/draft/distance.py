from dataclasses import dataclass

from draft.pose import PoseResult


@dataclass
class DistanceResult:
    """A result used when a distance estimate is requested.

    Used to return distance estimate results in a structured manner.
    
    Attributes:
        distance_m (float | None): Distance in meters or indication of no reliable estimate.
        off_signal (bool): A boolean to signal if the distance estimate is less than the minimum threshold.
    """
    distance_m: float | None  # None if landmarks not visible enough to estimate
    off_signal: bool


class DistanceEstimator:
    """Calculate distance estimates for person to camera.

    This class keeps track of the parameters needed to perform the estimate.
    It is used at each frame when the estimated distance needs to be recalculated.

    Attributes:
        user_height_m (float): The inputted user height in meters.
        threshold_m (float): The inputted minimum threshold that the user needs to have from the camera in meters.
        _focal_length (float): The calibrated camera focal length in meters.
    """
    def __init__(self, user_height_m: float, threshold_m: float) -> None:
        self.user_height_m = user_height_m
        self.threshold_m = threshold_m
        self._focal_length: float | None = None

    @property
    def is_calibrated(self) -> bool:
        return self._focal_length is not None

    def calibrate(self, pose_result: PoseResult, frame_height: int, reference_distance_m: float) -> bool:
        """Calibrates the estimator using PoseResult landmarks and inputted reference values.
        
        Args:
            pose_result (PoseResult): The PoseResult object with landmarks from MediaPipe.
            frame_height (int): Camera frame height in pixels.
            reference_distance_m (float): The distance of the user from the camera for calibration purposes.
        Returns:
            bool: Whether the calibration was successful or not.
        """

        # Calculate the height in pixels of the person.
        pixel_height = _landmark_pixel_height(pose_result, frame_height)

        # If no person was detected, return False.
        if pixel_height is None:
            return False
        
        # Calculate the focal length using the reference values and the person's computed pixel height.
        self._focal_length = (pixel_height * reference_distance_m) / self.user_height_m
        return True

    def estimate(self, pose_result: PoseResult, frame_height: int) -> DistanceResult:
        """Estimate the distance of the human to the camera.
        
        Args:
            pose_result (PoseResult): The object with landmarks from MediaPipe.
            frame_height (int): The camera frame's height.
        Returns:
            DistanceResult: The object with the estimated distance of the human to the camera and threshold violation signal.
        """

        # If calibration is not complete or PoseResult indicates NO PERSON, return an empty DistanceResult.
        if not self.is_calibrated or not pose_result.person_detected:
            return DistanceResult(distance_m=None, off_signal=False)

        # Compute the person's pixel height.
        pixel_height = _landmark_pixel_height(pose_result, frame_height)

        # Return an empty DistanceResult if no human pixel height could be computed or if focal length is not set.
        if pixel_height is None or self._focal_length is None:
            return DistanceResult(distance_m=None, off_signal=False)

        # Compute the estimated distance.
        distance = (self.user_height_m * self._focal_length) / pixel_height

        # Return a DistanceResult with the computed distance and use that distance to send the right off_signal.
        return DistanceResult(distance_m=distance, off_signal=distance < self.threshold_m)


def _landmark_pixel_height(pose_result: PoseResult, frame_height: int) -> float | None:
    """Compute the estimated person's height using PoseResult landmarks.
    
    Args:
        pose_result (PoseResult): The object with landmarks from MediaPipe for identified subjects in frame.
        frame_height (int): Camera frame height in pixels.

    Returns:
        float: The computed estimate for the person's height in the calibration camera frame.
    """

    # Return NO RESULT if the PoseResult indicates no person or if no landmarks are even present.
    if not pose_result.person_detected or not pose_result.landmarks:
        return None

    # Use landmarks with sufficient visibility to compute vertical span.
    visible_ys = [
        lm.y * frame_height
        for lm in pose_result.landmarks
        if lm.visibility > 0.5
    ]

    # Return NO RESULT if there are insufficient vertical measurements with good visibility.
    if len(visible_ys) < 4:
        return None

    # Compute the estimated human height using the difference between the max and min of the PoseResult landmarks.
    span = max(visible_ys) - min(visible_ys)
    return span if span > 0 else None

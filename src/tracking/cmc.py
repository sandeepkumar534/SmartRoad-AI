import cv2
import numpy as np

class CameraMotionCompensator:
    def __init__(self):
        self.prev_gray = None
        self.prev_keypoints = None
        self.max_corners = 500
        self.quality_level = 0.01
        self.min_distance = 10
        self.block_size = 3
        
        # Lucas Kanade optical flow parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

    def compute_affine_matrix(self, curr_frame):
        """Estimate the affine transform matrix from prev_frame to curr_frame."""
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is None:
            self.prev_gray = curr_gray
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, maxCorners=self.max_corners,
                qualityLevel=self.quality_level, minDistance=self.min_distance,
                blockSize=self.block_size
            )
            return None

        if self.prev_keypoints is None or len(self.prev_keypoints) < 10:
            # Need to re-detect features
            self.prev_keypoints = cv2.goodFeaturesToTrack(
                self.prev_gray, mask=None, maxCorners=self.max_corners,
                qualityLevel=self.quality_level, minDistance=self.min_distance,
                blockSize=self.block_size
            )
            if self.prev_keypoints is None or len(self.prev_keypoints) < 10:
                self.prev_gray = curr_gray
                return None

        # Calculate optical flow
        curr_keypoints, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_keypoints, None, **self.lk_params
        )

        if curr_keypoints is None or status is None:
            self.prev_gray = curr_gray
            self.prev_keypoints = None
            return None

        # Filter good points
        good_old = self.prev_keypoints[status == 1]
        good_new = curr_keypoints[status == 1]

        affine_matrix = None
        if len(good_new) >= 10:
            # Estimate affine transform
            affine_matrix, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

            if inliers is not None and np.sum(inliers) >= 10:
                # Keep inlier points for next frame tracking (faster than re-detecting all)
                self.prev_keypoints = good_new[inliers.flatten() == 1].reshape(-1, 1, 2)
            else:
                self.prev_keypoints = None
                affine_matrix = None
        else:
            self.prev_keypoints = None

        self.prev_gray = curr_gray
        return affine_matrix

    @staticmethod
    def apply_affine_to_bbox(bbox, affine_matrix):
        """Warp a bounding box (x1, y1, x2, y2) using the affine matrix."""
        if affine_matrix is None:
            return bbox
            
        x1, y1, x2, y2 = bbox
        # Represent corners as homogenous coordinates
        corners = np.array([
            [x1, y1, 1],
            [x2, y1, 1],
            [x2, y2, 1],
            [x1, y2, 1]
        ]).T
        
        # Apply transform
        warped_corners = affine_matrix @ corners
        
        # Get new bounding box
        new_x1 = np.min(warped_corners[0, :])
        new_y1 = np.min(warped_corners[1, :])
        new_x2 = np.max(warped_corners[0, :])
        new_y2 = np.max(warped_corners[1, :])
        
        return [new_x1, new_y1, new_x2, new_y2]

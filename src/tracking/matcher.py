import numpy as np
from scipy.optimize import linear_sum_assignment

def bb_iou(boxA, boxB):
    """Compute Intersection over Union of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

def compute_iou_distance(boxes1, boxes2):
    """Compute 1 - IoU distance matrix between two sets of boxes."""
    cost_matrix = np.ones((len(boxes1), len(boxes2)))
    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            cost_matrix[i, j] = 1.0 - bb_iou(box1, box2)
    return cost_matrix

def linear_assignment(cost_matrix, max_cost):
    """Perform Hungarian assignment based on a cost matrix and threshold."""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    matches = []
    unmatched_rows = []
    unmatched_cols = list(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] > max_cost:
            unmatched_rows.append(r)
        else:
            matches.append([r, c])
            unmatched_cols.remove(c)
            
    # Add rows that weren't part of the assignment
    for r in range(cost_matrix.shape[0]):
        if r not in row_ind:
            unmatched_rows.append(r)
            
    return np.array(matches), unmatched_rows, unmatched_cols

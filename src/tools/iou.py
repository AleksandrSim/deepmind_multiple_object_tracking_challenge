import numpy as np

from src.detect.utils import xyxy2xywh

low = np.s_[..., :2]
high = np.s_[..., 2:]


def iou_matrix(A: np.ndarray, B: np.ndarray):
    # Based on code from
    # https://stackoverflow.com/questions/57897578/efficient-way-to-calculate-all-ious-of-two-lists
    A = A[:, None].copy()
    B = B[None].copy()
    intrs = (np.maximum(0, np.minimum(A[high], B[high])
                        - np.maximum(A[low], B[low]))).prod(-1)
    return intrs / ((A[high]-A[low]).prod(-1)+(B[high]-B[low]).prod(-1)-intrs)


def merge_bboxes(A: np.ndarray, B: np.ndarray, threshold: float = 0.7)\
        -> np.ndarray:
    """Merge two sets of bboxes:

    1. If there are bboxes in B with iou > threshould, the bboxes in A are
        replaced with corresponding bboxes from B.
    2. If there are no corresponding bboxes in A, the bboxes from B are
        appended to A.

    Args:
        A (np.ndarray): A bboxes (N, P), where P >= 4 (x1, y1, x2, y2 and any
            extra values).
        B (np.ndarray): B bboxes (M, P).
        threshold (float, optional): IOU threshold, should be in [0.0..1.0].
            Defaults to 0.7.
    """
    iou = iou_matrix(A[:, :4], B[:, :4])
    similar_mask = iou > threshold

    # Find the row indices in B and A where IOUs > threshold
    a_indices, b_indices = np.where(similar_mask)

    # Update the corresponding rows in A with the rows from B
    print('Replaced in A:', a_indices, '<-', b_indices)
    A[a_indices] = B[b_indices]

    # Find rows in B that have no similar bboxes in A by the IOU threshold
    unique_b_indices = np.setdiff1d(np.arange(len(B)), b_indices)
    print('Added from B:', unique_b_indices)
    A = np.vstack((A, B[unique_b_indices]))

    return A


if __name__ == '__main__':
    A = np.array([[1, 1, 2, 2, 0.4],
                  [1, 1, 4, 4, 0.4],
                  [2, 2, 3, 4, 0.4]]).astype(float)
    B = np.array([[2, 2, 3, 3, 1.0],
                  [2, 2, 3.1, 4, 1.0]]).astype(float)
    ious = iou_matrix(A[:, :4], B[:, :4])
    max_indices = np.argmax(ious, axis=1)
    max_iou = np.max(ious, axis=1)
    print(ious)
    print(merge_bboxes(A, B, 0.7))

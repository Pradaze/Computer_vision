import numpy as np

def calculate_jaccard(pred_mask, gt_mask):
    # Calculate intersection and union directly with binary masks
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # Calculate Jaccard index
    jaccard = intersection / union if union > 0 else 0
    return jaccard


def calculate_dice(pred_mask, gt_mask):
    # Calculate true positives and sums directly with binary masks
    tp = np.logical_and(pred_mask, gt_mask).sum()
    pred_sum = pred_mask.sum()
    gt_sum = gt_mask.sum()

    # Calculate Dice coefficient
    dice = 2 * tp / (pred_sum + gt_sum) if (pred_sum + gt_sum) > 0 else 0
    return dice


def calculate_sensitivity_specificity(pred_mask, gt_mask):
    # Calculate TP, FP, FN, TN directly with binary masks
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()
    tn = np.logical_and(~pred_mask, ~gt_mask).sum()

    # Calculate sensitivity (recall) and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    return sensitivity, specificity


def calculate_precision_recall(pred_mask, gt_mask):
    # Calculate TP, FP, FN directly with binary masks
    tp = np.logical_and(pred_mask, gt_mask).sum()
    fp = np.logical_and(pred_mask, ~gt_mask).sum()
    fn = np.logical_and(~pred_mask, gt_mask).sum()

    # Calculate precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return precision, recall


from scipy.spatial.distance import directed_hausdorff


def calculate_hausdorff(pred_mask, gt_mask):
    # Get the coordinates of boundary pixels
    pred_boundary = np.argwhere(pred_mask)
    gt_boundary = np.argwhere(gt_mask)

    # Calculate Hausdorff distance
    if len(pred_boundary) > 0 and len(gt_boundary) > 0:
        hausdorff_dist = max(directed_hausdorff(pred_boundary, gt_boundary)[0],
                             directed_hausdorff(gt_boundary, pred_boundary)[0])
    else:
        hausdorff_dist = float('inf')
    return hausdorff_dist
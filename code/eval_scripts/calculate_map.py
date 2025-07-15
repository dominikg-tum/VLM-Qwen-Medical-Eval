import numpy as np
import supervision as sv
from supervision.metrics import MeanAveragePrecision
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def boxes_to_detections(boxes, class_ids=None):
    """
    Convert list of [x1, y1, x2, y2] boxes into a sv.Detections object.
    
    Args:
        boxes: List of boxes [[x1,y1,x2,y2], ...]
        class_ids: Optional list/array of class IDs for each box. 
                   If None, defaults to class_id=0 for all boxes.
    """
    if len(boxes) == 0:
        return sv.Detections.empty()
    
    xyxy = np.array(boxes, dtype=np.float32)
    
    if class_ids is None:
        class_id_arr = np.zeros(len(boxes), dtype=int)
    else:
        class_id_arr = np.array(class_ids, dtype=int)
        if len(class_id_arr) != len(boxes):
            raise ValueError("Length of class_ids must match number of boxes")
    
    confidence = np.ones(len(boxes), dtype=np.float32)  # fixed confidence = 1.0
    
    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id_arr,
        confidence=confidence
    )

def compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes):
    """
    Compute mAP metrics using supervision.
    
    Args:
        pred_boxes: list of predicted boxes [[x1, y1, x2, y2], ...]
        pred_classes: list of predicted class ids
        true_boxes: list of ground truth boxes [[x1, y1, x2, y2], ...]
        true_classes: list of ground truth class ids
    
    Returns:
        result: supervision.metrics.MeanAveragePrecisionResult
    """
    preds = boxes_to_detections(pred_boxes, pred_classes)
    targets = boxes_to_detections(true_boxes, true_classes)
    
    metric = MeanAveragePrecision()
    result = metric.update(preds, targets).compute()

    return result

def draw_boxes(pred_boxes, true_boxes, image=None, image_size=(1024, 1024), save_path=None):
    """
    Draw predicted and ground truth boxes on the given image or a blank canvas.

    Args:
        pred_boxes: list of predicted boxes [[x1, y1, x2, y2], ...]
        true_boxes: list of ground truth boxes [[x1, y1, x2, y2], ...]
        image: np.ndarray or str (image path). If None, uses blank canvas.
        image_size: tuple, used only if image is None.
    """
    if image is not None:
        if isinstance(image, str):
            img = plt.imread(image)
        else:
            img = image
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img)
        h, w = img.shape[:2]
    else:
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(np.ones((image_size[1], image_size[0], 3), dtype=np.uint8) * 240)
        w, h = image_size

    # Draw predicted boxes in red
    for box in pred_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

    # Draw ground truth boxes in green
    for box in true_boxes:
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=2, edgecolor='green', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)  # Flip y-axis (top-left origin)
    ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    pred_boxes = [
        [279.345, 482.761, 341.585, 543.730]
    ]
    pred_classes = [0]

    true_boxes = [
        [269.345, 500.761, 331.585, 543.730],
        [828.594, 700.222, 900.055, 800.994]
    ]
    true_classes = [0, 0]

    result = compute_map_supervision(pred_boxes, pred_classes, true_boxes, true_classes)
    # Example usage: draw_boxes(pred_boxes, true_boxes, image="path/to/image.png")
    draw_boxes(pred_boxes, true_boxes, image=None, image_size=(1024, 1024))

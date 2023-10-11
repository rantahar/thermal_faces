import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2


def display_image(temperature_image):
    plt.imshow(temperature_image, cmap='hot')
    plt.axis('off')
    plt.title('Temperature Image')
    plt.show()


def extract_subregions(array, labels=None, height=32, width=32, step_fraction=0.5):
    subregions = []
    array_height, array_width = array.shape
    step_size = int(min(height, width) * step_fraction)

    for y in range(0, array_height - height + 1, step_size):
        for x in range(0, array_width - width + 1, step_size):
            subregion = np.array(array[y:y+height, x:x+width])
            min_x = x+width*0.4
            min_y = y+height*0.3
            max_x = x+width*0.6
            max_y = y+height*0.6
            if labels is not None:
                contains_label = any(
                    label["x"] >= min_x and label["x"] < max_x and
                    label["y"] >= min_y and label["y"] < max_y
                    for label in labels
                )
            else:
                contains_label = None
            subregions.append((subregion, contains_label, x, y))

    return subregions


def extract_rescaled_subregions(image, labels, sizes, step_fraction=0.5):
    subregions = []
    smallest_size = min(sizes)
    
    resized_subregions = []
    for size in sizes:
        subregions += (extract_subregions(image, labels, size, size, step_fraction))
    
        # Resize each subregion
        for subregion, label, x, y in subregions:
            resized_subregion = cv2.resize(subregion, (smallest_size, smallest_size))
            resized_subregions.append((resized_subregion, label, x, y, size))
    
    return resized_subregions


def plot_boxes_on_image(image, boxes, labels = True):
    _, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')
    
    for box in boxes:
        rect = patches.Rectangle(
            (box[0], box[1]), box[2], box[3],
            linewidth=1, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
        
        if labels:
            ax.text(box[0], box[1], f'{box[4]:.2f}', bbox=dict(facecolor='white', alpha=0.2, edgecolor='none'))



    plt.show()


def calculate_iou(box1, boxes):
    intersection_x1 = np.maximum(box1[0], boxes[:, 0])
    intersection_y1 = np.maximum(box1[1], boxes[:, 1])
    intersection_x2 = np.minimum(box1[0] + box1[2], boxes[:, 0] + boxes[:, 2])
    intersection_y2 = np.minimum(box1[1] + box1[3], boxes[:, 1] + boxes[:, 3])

    intersection_width = np.maximum(0, intersection_x2 - intersection_x1)
    intersection_height = np.maximum(0, intersection_y2 - intersection_y1)

    intersection_areas = intersection_width * intersection_height
    box1_area = box1[2] * box1[3]
    box_areas = boxes[:, 2] * boxes[:, 3]

    iou = intersection_areas / (box1_area + box_areas - intersection_areas)
    return iou


def non_max_suppression(boxes, threshold):
    scores = np.array([b[4] for b in boxes])
    boxes = np.array(boxes)

    # Sort the scores in descending order
    sorted_indices = np.argsort(scores)[::-1]
    sorted_boxes = boxes[sorted_indices]
    sorted_scores = scores[sorted_indices]

    selected_boxes = []
    while len(sorted_boxes) > 0:
        # Select the box with the highest score
        best_box = sorted_boxes[0]
        selected_boxes.append(best_box)

        iou = calculate_iou(best_box, sorted_boxes)

        overlapping_indices = np.where(iou > threshold)[0]
        sorted_boxes = np.delete(sorted_boxes, overlapping_indices, axis=0)
        sorted_scores = np.delete(sorted_scores, overlapping_indices)

    return selected_boxes



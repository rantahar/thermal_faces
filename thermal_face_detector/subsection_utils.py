import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
import time
import torch


def display_image(temperature_image):
    plt.imshow(temperature_image, cmap='hot')
    plt.axis('off')
    plt.title('Temperature Image')
    plt.show()


def save_image_with_boxes(temperature_array, boxes, file_path):
    normalized_img = cv2.normalize(temperature_array, None, 0, 255, cv2.NORM_MINMAX)
    image = np.uint8(normalized_img)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for box in boxes:
        top_left = (int(box[0]), int(box[1]))
        bottom_right = (int(box[0] + box[2]), int(box[1] + box[3]))
        image = cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

        score = box[4]
        score_pos = (int(box[0]), int(box[1] - 10))
        cv2.putText(image, '{:.2f}'.format(score), score_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(file_path, image)


def check_label(labels, min_x, min_y, max_x, max_y):
    contains_label = any(
        label["x"] >= min_x and label["x"] < max_x and
        label["y"] >= min_y and label["y"] < max_y
        for label in labels
    )
    return contains_label

def extract_subregions(
    array, labels=None, height=32, width=32, step_fraction=0.5,
    require_forehead=True, require_nose=True
):
    subregions = []
    array_height, array_width = array.shape
    step_size = int(min(height, width) * step_fraction)

    foreheads = [l for l in labels if l['l'] == 1]
    noses = [l for l in labels if l['l'] == 3]

    for y in range(0, array_height - height + 1, step_size):
        for x in range(0, array_width - width + 1, step_size):
            subregion = np.array(array[y:y+height, x:x+width])
            contains_label = True
            if require_forehead and len(foreheads) > 0:
                contains_label = contains_label and check_label(
                    foreheads,x+width*0.2, y+height*0.2, x+width*0.8, y+height*0.5
                )
            if require_nose and len(noses) > 0:
                contains_label = contains_label and check_label(
                    noses,x+width*0.2, y+height*0.5, x+width*0.8, y+height*0.7
                )
            subregions.append((subregion, contains_label, x, y))

    return subregions


def extract_rescaled_subregions(
    image, labels, sizes, step_fraction=0.5, require_forehead=True, require_nose=True
):
    subregions = []
    smallest_size = min(sizes)
    
    resized_subregions = []
    for size in sizes:
        subregions += (extract_subregions(
            image, labels, size, size, step_fraction, require_forehead, require_nose
        ))
    
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


def box_iou(box1, box2):
    intersection_x1 = np.maximum(box1[0], box2[0])
    intersection_y1 = np.maximum(box1[1], box2[1])
    intersection_x2 = np.minimum(box1[0] + box1[2], box2[0] + box2[2])
    intersection_y2 = np.minimum(box1[1] + box1[3], box2[1] + box2[3])

    intersection_width = np.maximum(0, intersection_x2 - intersection_x1)
    intersection_height = np.maximum(0, intersection_y2 - intersection_y1)

    intersection_areas = intersection_width * intersection_height
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]

    iou = intersection_areas / (box1_area + box2_area - intersection_areas)
    return iou


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


def apply_to_matrix(model, matrix, sizes, step_fraction):
    region_size = model.image_width
    boxes = []
    for size in sizes:
        #start_time = time.time()
        scaled_x = int(matrix.shape[1]*region_size/size)
        scaled_y = int(matrix.shape[0]*region_size/size)
        resized = cv2.resize(matrix, (scaled_x, scaled_y))
        height, width = resized.shape
        step_size = int(region_size*step_fraction)
        for y in range(0, height - region_size + 1, step_size):
            for x in range(0, width - region_size + 1, step_size):
                subregion = np.array(resized[y:y+region_size, x:x+region_size])
                tensor = torch.tensor(subregion).unsqueeze(0)
                score = model(tensor).squeeze(0).item()
                boxes.append((x*size/region_size, y*size/region_size, size, size, score))

        #print(f"Size {size} sections processed in: {time.time() - start_time:.2f}s")
    
    return boxes


def apply_to_regions(model, matrix, step_fraction, threshold, regions, margin=16, max_overlap = 0.1):
    regions = non_max_suppression(regions, 0)

    boxes = []
    for region in regions:
        size = region[2]
        x1 = int(max(0, int(region[0]) - margin))
        x2 = int(min(matrix.shape[1], int(region[0]) + size + margin))
        y1 = int(max(0, int(region[1]) - margin))
        y2 = int(min(matrix.shape[0], int(region[1]) + size + margin))
        region_image = np.array(matrix[y1:y2, x1:x2])
        region_boxes = apply_to_matrix(model, region_image, [size], step_fraction)
        region_boxes = [
            (b[0] + x1, b[1] + y1, b[2], b[3], b[4])
            for b in region_boxes if b[4] > threshold
        ]
        boxes += non_max_suppression(region_boxes, max_overlap)

    boxes = non_max_suppression(boxes, max_overlap)
    return boxes


def scan_matrix(model, matrix, scan_size, scan_margin = None, scan_step = 0.3, threshold = 0):
    if scan_margin is None:
        scan_margin = scan_size//8

    regions_of_interest = apply_to_matrix(model, matrix, [scan_size], scan_step)
    regions_of_interest = [b for b in regions_of_interest if b[4] > threshold]
    regions_of_interest = non_max_suppression(regions_of_interest, 0)
    print(f"{len(regions_of_interest)} potential regions")
    print(regions_of_interest)

    return regions_of_interest


def scan_and_apply(model, matrix, sizes, step_fraction, threshold, scan_size = None, scan_margin = 8, scan_step = 0.3, max_overlap = 0.1):
    if scan_size is None:
        scan_size = max(sizes)

    regions_of_interest = scan_matrix(model, matrix, scan_size, scan_margin, scan_step)
    boxes = apply_to_regions(model, matrix, step_fraction, threshold, regions_of_interest, scan_margin, max_overlap)

    return boxes




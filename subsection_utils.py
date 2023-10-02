import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


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
            max_y = y+height*0.5
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


def plot_boxes_on_image(image, boxes):
    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    ax.imshow(image, cmap='gray')
    
    # Add bounding boxes to the image
    for box in boxes:
        rect = patches.Rectangle((box[0], box[1]), box[2], box[3], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Show plot
    plt.show()

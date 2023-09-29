import numpy as np
import matplotlib.pyplot as plt


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
            if labels is not None:
                contains_label = any(
                    label["x"] >= x and label["x"] < x+width and
                    label["y"] >= y and label["y"] < y+height
                    for label in labels
                )
            else:
                contains_label = None
            subregions.append((subregion, contains_label, x, y))

    return subregions
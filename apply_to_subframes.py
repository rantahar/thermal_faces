import sys
import click
import torch
import numpy as np
from subsection_utils import plot_boxes_on_image, non_max_suppression
import cv2
import time

from reduce_model import FaceDetector


@click.command()
@click.option("--threshold", default=0.99, help="Threshold for detecting a face in a bounding box.")
@click.option("--region_sizes", default=[32,48,64], help="Size of the bounding box.", multiple=True)
@click.option("--step_fraction", default=0.2, help="Step between bounding boxes as a fraction of its size.")
@click.option("--image_file", help="Path to an image file.", required=True)
@click.option("--model_file", help="Path to the model file.", required=True)
def apply_to_frame(threshold, region_sizes, step_fraction, image_file, model_file):
    # Load the model from disk
    model_dict = torch.load(
        model_file,
        map_location=torch.device('cpu')
    )
    units = model_dict.get("units")
    region_size = model_dict.get("region_size")

    model = FaceDetector(region_size, region_size, units)
    model.load_state_dict(model_dict['model_state_dict'])
    model.eval()

    # Load temperature values and process
    image = np.load(image_file).astype(np.float32)

    boxes = []
    height, width = image.shape
    for size in region_sizes:
        start_time = time.time()
        step_size = int(size*step_fraction)
        for y in range(0, height - size + 1, step_size):
            for x in range(0, width - size + 1, step_size):
                subregion = np.array(image[y:y+size, x:x+size])
                subregion = cv2.resize(subregion, (region_size, region_size))
                tensor = torch.tensor(subregion).unsqueeze(0)
                score = model(tensor).squeeze(0).item()
                if score > threshold:
                    boxes.append((x, y, size, size, score))

        print(f"Size {size} sections processed in: {time.time() - start_time:.2f}s")


    boxes = non_max_suppression(boxes, 0.1)

    plot_boxes_on_image(image, boxes)



if __name__ == '__main__':
    apply_to_frame()

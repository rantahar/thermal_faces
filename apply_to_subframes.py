import sys
import click
import torch
import numpy as np
from subsection_utils import plot_boxes_on_image, non_max_suppression
import cv2
import time

from reduce_model import FaceDetector


@click.command()
@click.option("--threshold", default=0, help="Threshold for detecting a face in a bounding box.")
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
    model.train()

    # Load temperature values and process
    image = np.load(image_file).astype(np.float32)

    boxes = []
    for size in region_sizes:
        start_time = time.time()
        scaled_x = int(image.shape[1]*region_size/size)
        scaled_y = int(image.shape[0]*region_size/size)
        resized = cv2.resize(image, (scaled_x, scaled_y))
        height, width = resized.shape
        step_size = int(region_size*step_fraction)
        for y in range(0, height - region_size + 1, step_size):
            for x in range(0, width - region_size + 1, step_size):
                subregion = np.array(resized[y:y+region_size, x:x+region_size])
                tensor = torch.tensor(subregion).unsqueeze(0)
                score = model(tensor).squeeze(0).item()
                if score > threshold:
                    boxes.append((x*size/region_size, y*size/region_size, size, size, score))

        print(f"Size {size} sections processed in: {time.time() - start_time:.2f}s")


    boxes = non_max_suppression(boxes, 0.1)

    plot_boxes_on_image(image, boxes)



if __name__ == '__main__':
    apply_to_frame()

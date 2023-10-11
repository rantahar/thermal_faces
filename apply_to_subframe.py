import click
import torch
import numpy as np
from thermal_face_detector.subsection_utils import plot_boxes_on_image, non_max_suppression, apply_to_matrix
from thermal_face_detector.reduce_model import FaceDetector


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

    regions_of_interest = apply_to_matrix(model, image, [64], 0.2)
    regions_of_interest = [b for b in regions_of_interest if b[4] > 0]
    regions_of_interest = non_max_suppression(regions_of_interest, 0)
    plot_boxes_on_image(image, regions_of_interest)

    boxes = []
    for region in regions_of_interest:
        x1 = max(0, int(region[0]) - 32)
        x2 = min(image.shape[1], int(region[0]) + 64 + 32)
        y1 = max(0, int(region[1]) - 32)
        y2 = min(image.shape[0], int(region[1]) + 64 + 32)
        region_image = np.array(image[y1:y2, x1:x2])
        region_boxes = apply_to_matrix(model, region_image, region_sizes, step_fraction)
        region_boxes = [
            (b[0] + x1, b[1] + y1, b[2], b[3], b[4])
            for b in region_boxes if b[4] > threshold
        ]
        boxes += non_max_suppression(region_boxes, 0.1)
        #plot_boxes_on_image(image, boxes)
        
    boxes = non_max_suppression(boxes, 0.1)

    plot_boxes_on_image(image, boxes)



if __name__ == '__main__':
    apply_to_frame()

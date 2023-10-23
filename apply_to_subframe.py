import click
import torch
import time
import numpy as np
from thermal_face_detector.subsection_utils import plot_boxes_on_image, scan_and_apply
from thermal_face_detector.reduce_model import FaceDetector

@click.command()
@click.option("--threshold", default=0, help="Threshold for detecting a face in a bounding box.")
@click.option("--region_sizes", default=[32,48,64], help="Size of the bounding box.", multiple=True)
@click.option("--step_fraction", default=0.25, help="Step between bounding boxes as a fraction of its size.")
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

    start_time = time.time()
    boxes = scan_and_apply(model, image, region_sizes, step_fraction, threshold, max_overlap = 0.1)
    print(time.time() - start_time)
    plot_boxes_on_image(image, boxes)



if __name__ == '__main__':
    apply_to_frame()

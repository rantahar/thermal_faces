import click
import torch
import numpy as np

from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames
from thermal_face_detector.subsection_utils import plot_boxes_on_image, non_max_suppression, apply_to_matrix
from thermal_face_detector.reduce_model import FaceDetector


@click.command()
@click.option("--threshold", default=0, help="Threshold for detecting a face in a bounding box.")
@click.option("--region_sizes", default=[32,48,64], help="Size of the bounding box.", multiple=True)
@click.option("--step_fraction", default=0.2, help="Step between bounding boxes as a fraction of its size.")
@click.option("--video_file", help="Path to a video file.", required=True)
@click.option("--model_file", help="Path to the model file.", required=True)
def run(threshold, region_sizes, step_fraction, video_file, model_file):
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

    # Load the model
    metadata = extract_metadata(video_file)
    width = int(metadata["Raw Thermal Image Width"])
    height = int(metadata["Raw Thermal Image Height"])
    bitdepth = 16
    frame_size = width * height * (bitdepth // 8)

    print(frame_size)
    
    # Process the frames
    for frame_index, frame in enumerate(seq_frames(video_file)):
        raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width).astype(np.float32)
        temperature = convert_to_temperature(raw_data, metadata)


        boxes = apply_to_matrix(model, temperature, region_sizes, step_fraction)
        boxes = [b for b in boxes if b[4] > threshold]
        boxes = non_max_suppression(boxes, 0.1)


        plot_boxes_on_image(temperature, boxes)


if __name__ == '__main__':
    run()


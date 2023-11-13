import click
import torch
import numpy as np

from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames
from thermal_face_detector.subsection_utils import plot_boxes_on_image, box_iou, apply_to_regions, scan_and_apply, non_max_suppression
from thermal_face_detector.reduce_model import FaceDetector


@click.command()
@click.option("--new_region_threshold", default=2, help="Threshold for detecting a new face in a bounding box.")
@click.option("--update_threshold", default=2, help="Threshold for detecting a face in the next frame.")
@click.option("--region_sizes", default=[32,48,64], help="Size of the bounding box.", multiple=True)
@click.option("--step_fraction", default=0.2, help="Step between bounding boxes as a fraction of its size.")
@click.option("--max_overlap", default=0.1, help="Maximum overlap (intersection over union) allowed between boxes.")
@click.option("--video_file", help="Path to a video file.", required=True)
@click.option("--model_file", help="Path to the model file.", required=True)
def run(new_region_threshold, update_threshold, region_sizes, step_fraction, max_overlap, video_file, model_file):
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

    # Process the frames, keeping track of boxes
    boxes = []
    next_box_index = 0
    for frame_index, frame in enumerate(seq_frames(video_file)):
        raw_data = np.frombuffer(frame[len(frame)-frame_size:], dtype=np.uint16).reshape(height, width).astype(np.float32)
        temperature = convert_to_temperature(raw_data, metadata,)

        # Scan for new faces every 100 frames, otherwise just
        # check existing boxes
        regions = [(box["x"], box["y"], box["width"], box["height"], box["score"]) for box in boxes]
        new_boxes = apply_to_regions(model, temperature, region_sizes, step_fraction, update_threshold, regions)
        print(f"new {len(new_boxes)}")

        if frame_index == 0 or (frame_index+1)%100 == 0:
            new_boxes += scan_and_apply(model, temperature, region_sizes, step_fraction, new_region_threshold)
            new_boxes = non_max_suppression(new_boxes, max_overlap)

        for box in new_boxes:
            # identify the new boxes with existing ones
            found_box = False
            for i, old_box in enumerate(boxes):
                old_box = (old_box["x"], old_box["y"], old_box["width"], old_box["height"])
                print()
                print(box)
                print(old_box)
                print(box_iou(box, old_box))
                if box_iou(box, old_box) > max_overlap:
                    # replace old box
                    boxes[i]["x"] = box[0]
                    boxes[i]["y"] = box[1]
                    boxes[i]["width"] = box[2]
                    boxes[i]["height"] = box[3]
                    boxes[i]["score"] = box[4]
                    found_box = True
                    break
            if not found_box:
                # new box
                boxes.append({"x": box[0], "y": box[1], "width": box[2], "height": box[3], "score": box[4], "index": next_box_index})
                next_box_index += 1

        if len(boxes) > 0:
            print()
            for box in boxes:
                print(box)

        if frame_index == 0 or (frame_index+1)%100 == 0:
            regions = [(box["x"], box["y"], box["width"], box["height"], box["score"]) for box in boxes]
            plot_boxes_on_image(temperature, regions)


if __name__ == '__main__':
    run()




import os
import csv
import click
import torch
import numpy as np

from thermal_faces.seq_reader import extract_metadata, convert_to_temperature, seq_frames
from thermal_face_detector.subsection_utils import plot_boxes_on_image, box_iou, apply_to_regions, scan_and_apply, non_max_suppression, save_image_with_boxes
from thermal_face_detector.reduce_model import FaceDetector


def append_boxes_to_csv(boxes, file_path):
    # Check if the CSV file needs headers
    file_exists = os.path.isfile(file_path)
    
    # Open the file in 'append' mode
    with open(file_path, 'a', newline='') as f:
        # Create a CSV writer object
        writer = csv.DictWriter(f, fieldnames=boxes[0].keys())

        # Write the header only if the file is created for the first time
        if not file_exists:
            writer.writeheader()

        # Write the rows
        for box in boxes:
            writer.writerow(box)



@click.command()
@click.option("--new_region_threshold", default=10, help="Threshold for detecting a new face in a bounding box.")
@click.option("--update_threshold", default=5, help="Threshold for detecting a face in the next frame.")
@click.option("--region_sizes", default=[32,48,64], help="Size of the bounding box.", multiple=True)
@click.option("--step_fraction", default=0.1, help="Step between bounding boxes as a fraction of its size.")
@click.option("--max_overlap", default=0.1, help="Maximum overlap (intersection over union) allowed between boxes.")
@click.option("--scan_every", default=1000, help="Number of frames between scanning for new faces.")
@click.option("--video_file", help="Path to a video file.", required=True)
@click.option("--model_file", help="Path to the model file.", required=True)
@click.option("--output_file", default=None, help="Output csv file.")
@click.option("--frame_images_dir", default="frame_images", help="Folder for saving frame images")
def run(new_region_threshold, update_threshold, region_sizes, step_fraction, max_overlap, scan_every, video_file, model_file, output_file, frame_images_dir):
    if not os.path.exists(frame_images_dir):
        os.makedirs(frame_images_dir)
    
    if output_file is None:
        output_file = os.path.splitext(os.path.basename(video_file))[0] + ".csv"

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
        new_boxes = apply_to_regions(model, temperature, step_fraction, update_threshold, regions)
        #print(f"new {len(new_boxes)}")

        if frame_index == 0 or frame_index%scan_every == 0:
            print("Scanning for new faces...")
            new_boxes += scan_and_apply(model, temperature, region_sizes, step_fraction, new_region_threshold)
            new_boxes = non_max_suppression(new_boxes, max_overlap)

        frame_boxes = []
        for box in new_boxes:
            box_dict = {
                "x": box[0], "y": box[1],
                "width": box[2], "height": box[3],
                "score": box[4], "frame": frame_index
            }
            # identify the new boxes with existing ones
            found_box = False
            for i, old_box in enumerate(boxes):
                old_box = (
                    old_box["x"], old_box["y"],
                    old_box["width"], old_box["height"]
                )
                print()
                print(box)
                print(old_box)
                print(box_iou(box, old_box))
                if box_iou(box, old_box) > max_overlap:
                    # replace old box
                    box_dict["index"] = boxes[i]["index"]
                    boxes[i] = box_dict
                    found_box = True
                    break
            if not found_box:
                # new box
                box_dict["index"] = next_box_index
                boxes.append(box_dict)
                next_box_index += 1
            frame_boxes.append(box_dict)

        if len(boxes) > 0:
            print(boxes)
        if len(frame_boxes) > 0:
            append_boxes_to_csv(frame_boxes, output_file)

        if frame_index == 0 or frame_index%scan_every == 0:
            regions = [(box["x"], box["y"], box["width"], box["height"], box["score"]) for box in boxes]
            file_name = f"{os.path.basename(video_file).split('.')[0]}_{frame_index}.png"
            file_name = os.path.join(frame_images_dir, file_name)
            save_image_with_boxes(temperature, regions, file_name)


if __name__ == '__main__':
    run()




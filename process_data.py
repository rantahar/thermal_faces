import os
import numpy as np
import json
import torch
import random
import click

from thermal_face_detector.subsection_utils import extract_training_data_with_nose


if not os.path.exists("saved"):
    os.makedirs("saved")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)


def batch_data(data, batch_size):

    random.shuffle(data)
    data = torch.tensor(np.array(data, dtype=np.float32), dtype=torch.float32)
    region_size = data.shape[1]
    n_batches = data.shape[0]//batch_size
    data = data[:n_batches*batch_size]
    batches = torch.reshape(data, [-1, batch_size, region_size, region_size])

    return batches


def get_subregions_by_label(data, region_sizes, step_fraction, keep_fraction):
    data_positive = []
    data_negative = []

    for resolution in data:
        for frame_index, array, json_data in data[resolution]:
            if len(json_data) == 0:
                continue
            
            regions = extract_training_data_with_nose(
                array, json_data, region_sizes, require_forehead=True
            )
            for region in regions:
                if region[1]:
                    data_positive.append(region[0])
                else:
                    #if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                    data_negative.append(region[0])
    
    return data_positive, data_negative


def load_npy_files(data_path, batch_size, region_sizes, step_fraction, keep_fraction, validation_fraction):
    data = {}
    
    for file_name in os.listdir(data_path):
        if file_name.endswith('.npy'):
            words = file_name.split('_')
            frame_index = int(words[-1].replace(".npy",""))
            video_name = "_".join(words[:-1])
            file_path = os.path.join(data_path, file_name)

            array = np.load(file_path)
            json_file_path = os.path.join(data_path, file_name.replace(".npy",".json"))
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)

            file_path = os.path.join(data_path, file_name)
            if video_name in data:
                data[video_name].append((frame_index, array, json_data["labels"]))
            else:
                data[video_name] = [
                    (frame_index, array, json_data["labels"]),
                ]

    train_data_by_resolution = {}
    valid_data_by_resolution = {}

    for video_name in data.keys():
        data[video_name].sort(key=lambda x: x[0])

    for video_name, frames in data.items():
        if len(frames) > 2:
            num_frames = len(frames)
            resolution = frames[0][1].shape
            start_frame = int(np.floor((0.5-0.5*validation_fraction)*num_frames))
            end_frame = int(np.ceil((0.5+0.5*validation_fraction)*num_frames))

            valid_frames = frames[start_frame:end_frame]
            train_frames = frames[:start_frame] + frames[end_frame:]

            if resolution not in train_data_by_resolution:
                train_data_by_resolution[resolution] = train_frames
            else:
                train_data_by_resolution[resolution] += train_frames

            if resolution not in valid_data_by_resolution:
                valid_data_by_resolution[resolution] = valid_frames
            else:
                valid_data_by_resolution[resolution] += valid_frames

    print("Creating subregions")
    train_data_positive, train_data_negative = get_subregions_by_label(
        train_data_by_resolution, region_sizes, step_fraction, keep_fraction
    )
    valid_data_positive, valid_data_negative = get_subregions_by_label(
        valid_data_by_resolution, region_sizes, step_fraction, keep_fraction
    )


    print("Creating batches")
    train_data_positive = batch_data(train_data_positive, batch_size)
    train_data_negative = batch_data(train_data_negative, batch_size)
    valid_data_positive = batch_data(valid_data_positive, batch_size)
    valid_data_negative = batch_data(valid_data_negative, batch_size)
    
    return train_data_positive, train_data_negative, valid_data_positive, valid_data_negative




@click.command()
@click.option("--data_path", help="Path to the data folder.", required=True)
@click.option("--batch_size", default=100, help="Size of the training batches.")
@click.option("--region_sizes", default=[32,48,64], help="List of bounding box sizes.", multiple=True)
@click.option("--step_fraction", default=0.1, help="Step between bounding boxes as a fraction of its size.")
@click.option("--keep_fraction", default=0.01, help="Fraction of negative data to keep.")
@click.option("--validation_fraction", default=0.1, help="Fraction of negative data to keep.")
def preprocess_images(data_path, batch_size, region_sizes, step_fraction, keep_fraction, validation_fraction):

    train_data_positive, train_data_negative, valid_data_positive, valid_data_negative = load_npy_files(data_path, batch_size, region_sizes, step_fraction, keep_fraction, validation_fraction)

    print(f"Positive training data: {train_data_positive.shape}, negative: {train_data_negative.shape}")
    print(f"Positive validation data: {valid_data_positive.shape}, negative: {valid_data_negative.shape}")

    torch.save(train_data_positive, 'train_data_positive.pt')
    torch.save(train_data_negative, 'train_data_negative.pt')
    torch.save(valid_data_positive, 'valid_data_positive.pt')
    torch.save(valid_data_negative, 'valid_data_negative.pt')



if __name__ == '__main__':
    preprocess_images()



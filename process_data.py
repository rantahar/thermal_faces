import os
import numpy as np
import json
import torch
import random

from subsection_utils import extract_rescaled_subregions

data_path = "data/train_data"
batch_size = 100
region_sizes = [32, 48, 64]
region_step_fraction = 0.1
keep_fraction = 0.01
label_keep_fraction = 1

region_size = min(region_sizes)

if not os.path.exists("saved"):
    os.makedirs("saved")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


def batch_data(data):
    random.shuffle(data)
    data = torch.tensor(np.array(data, dtype=np.float32), dtype=torch.float32)
    n_batches = data.shape[0]//batch_size
    data = data[:n_batches*batch_size]
    batches = torch.reshape(data, [-1, batch_size, region_size, region_size])

    return batches


def load_npy_files(folder_path, validation_fraction=0.1):
    data = {}
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            words = file_name.split('_')
            frame_index = int(words[-1].replace(".npy",""))
            video_name = "_".join(words[:-1])
            file_path = os.path.join(folder_path, file_name)

            array = np.load(file_path)
            json_file_path = os.path.join(folder_path, file_name.replace(".npy",".json"))
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)

            file_path = os.path.join(folder_path, file_name)
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

    train_data_positive = []
    train_data_negative = []
    valid_data_positive = []
    valid_data_negative = []

    print("Creating subregions")
    for resolution in train_data_by_resolution:
        for frame_index, array, json_data in train_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]
            if len(labels) == 0:
                continue

            regions = extract_rescaled_subregions(
                array, labels, [32, 48, 64], region_step_fraction
            )
            for region in regions:
                if region[1]:
                    if np.random.choice([True, False], 1, p=[label_keep_fraction,1-label_keep_fraction]):
                        train_data_positive.append(region[0])
                else:
                    if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                        train_data_negative.append(region[0])


    for resolution in valid_data_by_resolution:
        for frame_index, array, json_data in valid_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]
            if len(labels) == 0:
                continue

            regions = extract_rescaled_subregions(
                array, labels, [32, 48, 64], region_step_fraction
            )
            for region in regions:
                if region[1]:
                    if np.random.choice([True, False], 1, p=[label_keep_fraction,1-label_keep_fraction]):
                        valid_data_positive.append(region[0])
                else:
                    if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                        valid_data_negative.append(region[0])


    print("Creating batches")
    train_data_positive = batch_data(train_data_positive)
    train_data_negative = batch_data(train_data_negative)
    valid_data_positive = batch_data(valid_data_positive)
    valid_data_negative = batch_data(valid_data_negative)    
    
    return train_data_positive, train_data_negative, valid_data_positive, valid_data_negative



train_data_positive, train_data_negative, valid_data_positive, valid_data_negative = load_npy_files(data_path)

print(f"Positive training data: {train_data_positive.shape}, negative: {train_data_negative.shape}")
print(f"Positive validation data: {valid_data_positive.shape}, negative: {valid_data_negative.shape}")

torch.save(train_data_positive, 'train_data_positive.pt')
torch.save(train_data_negative, 'train_data_negative.pt')
torch.save(valid_data_positive, 'valid_data_positive.pt')
torch.save(valid_data_negative, 'valid_data_negative.pt')


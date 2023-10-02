import os
import numpy as np
import json
import torch
import torch.nn as nn
import time

from subsection_utils import extract_subregions, plot_boxes_on_image
from reduce_model import FaceDetector

data_path = "data/train_data"
batch_size = 50
learning_rate = 1e-4
region_size = 48
region_step_fraction = 0.1
keep_fraction = 0.01
label_keep_fraction = 1
save_every = 1
num_epochs = 10001
units = 16

print("units:", units)

save_path = f"saved/reduction_model_{units}_{region_size}"

if not os.path.exists("saved"):
    os.makedirs("saved")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


def batch_data(data):
    positive_arrays = []
    negative_arrays = []
    positive_batches = []
    negative_batches = []

    for i, (array, label, _ , _) in enumerate(data):
        if label:
            positive_arrays.append(array)
        else:
            negative_arrays.append(array)

        if len(positive_arrays) == batch_size or i == (len(data)-1):
            positive_arrays_tensor = torch.tensor(np.array(positive_arrays), dtype=torch.float32).to(device)
            positive_batches.append(positive_arrays_tensor)
            positive_arrays = []

        if len(negative_arrays) == batch_size or i == (len(data)-1):
            negative_arrays_tensor = torch.tensor(np.array(negative_arrays), dtype=torch.float32).to(device)
            negative_batches.append(negative_arrays_tensor)
            negative_arrays = []


    return positive_batches, negative_batches



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

    train_data = []
    valid_data = []

    print("Creating subregions")
    for resolution in train_data_by_resolution:
        for frame_index, array, json_data in train_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]
            #labels = [l for l in json_data]
            if len(labels) == 0:
                continue

            regions = extract_subregions(
                array, labels, region_size, region_size, region_step_fraction
            )
            for region in regions:
                if region[1]:
                    if np.random.choice([True, False], 1, p=[label_keep_fraction,1-label_keep_fraction]):
                        train_data.append(region)
                else:
                    if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                        train_data.append(region)

            #boxes = [(r[2], r[3], region_size, region_size) for r in regions if r[1] > 0.5]
            #plot_boxes_on_image(array, boxes)


    for resolution in valid_data_by_resolution:
        for frame_index, array, json_data in valid_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]
            #labels = [l for l in json_data]
            if len(labels) == 0:
                continue

            regions = extract_subregions(
                array, labels, region_size, region_size, region_step_fraction
            )
            for region in regions:
                if region[1]:
                    if np.random.choice([True, False], 1, p=[label_keep_fraction,1-label_keep_fraction]):
                        valid_data.append(region)
                else:
                    if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                        valid_data.append(region)


    print("Creating batches")
    train_data_positive, train_data_negative = batch_data(train_data)
    valid_data_positive, valid_data_negative = batch_data(valid_data)
            
    return (train_data_positive, train_data_negative), (valid_data_positive, valid_data_negative)



train_data, valid_data = load_npy_files(data_path)
print(f"Positive training data: {len(train_data[0])}, negative: {len(train_data[1])}")
print(f"Negative validation data: {len(valid_data[0])}, negative: {len(valid_data[1])}")



loss_function = nn.BCEWithLogitsLoss()

def compute_loss(positive, negative):
    loss = loss_function(positive, torch.ones_like(positive))
    loss += loss_function(negative, torch.zeros_like(negative))
    return loss


def calculate_accuracy(positive, negative):
    positive = nn.functional.sigmoid(positive)
    negative = nn.functional.sigmoid(negative)
    return 0.5*(positive.mean().item() + (1-negative.mean().item()))


faceDetector = FaceDetector(region_size, region_size, units).to(device)
optimizer = torch.optim.Adam(faceDetector.parameters(), lr=learning_rate)


import itertools
negative_train_data = itertools.cycle(train_data[1])
negative_valid_data = itertools.cycle(valid_data[1])



for epoch in range(num_epochs):
    faceDetector.train()
    batch_num = 0
    train_loss = 0
    train_accuracy = 0
    std = 0
    start_time = time.time()
    for i, data in enumerate(train_data[0]):
        positive_data = data
        negative_data = next(negative_train_data)

        positive_predictions = faceDetector(positive_data)
        negative_predictions = faceDetector(negative_data)
        loss = compute_loss(positive_predictions, negative_predictions)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        std += positive_predictions.detach().numpy().std() + negative_predictions.detach().numpy().std()
        train_accuracy += calculate_accuracy(positive_predictions, negative_predictions)
        batch_num += 1

    batch_time = time.time() - start_time
    train_loss /= batch_num
    train_accuracy /= batch_num
    std /= batch_num
    print(f"Epoch {epoch}: Loss = {train_loss}, Accuracy = {train_accuracy}, std = {std}, Time: {batch_time:.2f}s")

    validation_loss = 0
    validation_accuracy = 0
    std = 0
    total = 0
    with torch.no_grad():
        for data in valid_data[0]:
            positive_data = data
            negative_data = next(negative_valid_data)

            positive_predictions = faceDetector(positive_data)
            negative_predictions = faceDetector(negative_data)
            loss = compute_loss(positive_predictions, negative_predictions)
            validation_loss += loss.item()

            std += positive_predictions.detach().numpy().std() + negative_predictions.detach().numpy().std()
            validation_accuracy += calculate_accuracy(positive_predictions, negative_predictions)
            total += 1

        validation_loss /= total
        validation_accuracy /= total
        std /= total
        print(f"Validation: Loss = {validation_loss}, Accuracy = {validation_accuracy}, std = {std}", flush=True)


    if epoch%save_every == 0:
        model_state = faceDetector.state_dict()
        torch.save(
            {
                'model_state_dict': faceDetector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'units': units,
                'region_size': region_size
            },
            f"{save_path}_{epoch}"
        )
        with open(f"{save_path}_{epoch}.json", 'w') as f:
            json.dump({
                'epoch': epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }, f, indent=4)

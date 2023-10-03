import os
import numpy as np
import json
import torch
import torch.nn as nn
import time
import random
import itertools

from subsection_utils import extract_subregions, plot_boxes_on_image
from reduce_model import FaceDetector

data_path = "data/train_data"
batch_size = 100
learning_rate = 1e-5
region_size = 48
region_step_fraction = 0.1
keep_fraction = 0.01
label_keep_fraction = 1
negatives_per_positive = 5
save_every = 100
num_epochs = 10001
units = 8

print("units:", units)

save_path = f"saved/reduction_model_{units}_{region_size}"

if not os.path.exists("saved"):
    os.makedirs("saved")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


def batch_data(data):
    arrays = []
    batches = []

    for i, array in enumerate(data):
        arrays.append(array)
        if len(arrays) == batch_size or i == (len(data)-1):
            arrays_tensor = torch.tensor(np.array(arrays), dtype=torch.float32).to(device)
            batches.append(arrays_tensor)
            arrays = []

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

            regions = extract_subregions(
                array, labels, region_size, region_size, region_step_fraction
            )
            for region in regions:
                if region[1]:
                    if np.random.choice([True, False], 1, p=[label_keep_fraction,1-label_keep_fraction]):
                        train_data_positive.append(region[0])
                else:
                    if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                        train_data_negative.append(region[0])

            #boxes = [(r[2], r[3], region_size, region_size) for r in regions if r[1] > 0.5]
            #plot_boxes_on_image(array, boxes)


    for resolution in valid_data_by_resolution:
        for frame_index, array, json_data in valid_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]
            if len(labels) == 0:
                continue

            regions = extract_subregions(
                array, labels, region_size, region_size, region_step_fraction
            )
            for region in regions:
                if region[1]:
                    if np.random.choice([True, False], 1, p=[label_keep_fraction,1-label_keep_fraction]):
                        valid_data_positive.append(region[0])
                else:
                    if np.random.choice([True, False], 1, p=[keep_fraction,1-keep_fraction]):
                        valid_data_negative.append(region[0])


    print("Creating batches")
    random.shuffle(train_data_positive)
    random.shuffle(train_data_negative)
    random.shuffle(valid_data_positive)
    random.shuffle(valid_data_negative)
    train_data_positive = batch_data(train_data_positive)
    train_data_negative = batch_data(train_data_negative)
    valid_data_positive = batch_data(valid_data_positive)
    valid_data_negative = batch_data(valid_data_negative)
    
    return (train_data_positive, train_data_negative), (valid_data_positive, valid_data_negative)



train_data, valid_data = load_npy_files(data_path)
print(f"Positive training data: {len(train_data[0])}, negative: {len(train_data[1])}")
print(f"Positive validation data: {len(valid_data[0])}, negative: {len(valid_data[1])}")



faceDetector = FaceDetector(region_size, region_size, units).to(device)
optimizer = torch.optim.Adam(faceDetector.parameters(), lr=learning_rate)
loss_function = nn.BCEWithLogitsLoss()

# Cyclical iterator to get batches of negative examples. Helps in balancing
# positive and negative examples when there are many more negative ones.
negative_train_data = itertools.cycle(train_data[1])


for epoch in range(num_epochs):
    faceDetector.train()
    negative_batches = 0
    positive_batches = 0
    train_loss = 0
    train_accuracy_positive = 0
    train_accuracy_negative = 0
    start_time = time.time()
    for i, data in enumerate(train_data[0]):
        positive_predictions = faceDetector(data)
        loss = loss_function(positive_predictions, torch.ones_like(positive_predictions))
        train_accuracy_positive += nn.functional.sigmoid(positive_predictions).mean().item()

        for i in range(negatives_per_positive):
            negative_data = next(negative_train_data)
            negative_predictions = faceDetector(negative_data)
            loss += loss_function(negative_predictions, torch.zeros_like(negative_predictions))
            train_accuracy_negative += 1 - nn.functional.sigmoid(negative_predictions).mean().item()
            negative_batches += 1

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        positive_batches += 1

    batch_time = time.time() - start_time
    train_loss /= negative_batches + positive_batches
    train_accuracy_positive /= positive_batches
    train_accuracy_negative /= negative_batches
    print(f"Epoch {epoch}: Loss = {train_loss}, Accuracy positive = {train_accuracy_positive}, Accuracy negative {train_accuracy_negative}, Time: {batch_time:.2f}s")

    validation_loss = 0
    validation_accuracy_positive = 0
    validation_accuracy_negative = 0
    total_positive = 0
    total_negative = 0
    with torch.no_grad():
        for data in valid_data[0]:
            predictions = faceDetector(data)
            validation_loss += loss_function(predictions, torch.ones_like(predictions)).item()
            validation_accuracy_positive += nn.functional.sigmoid(predictions).mean().item()
            total_positive += 1
        
        for data in valid_data[1]:
            predictions = faceDetector(data)
            validation_loss += loss_function(predictions, torch.zeros_like(predictions)).item()
            validation_accuracy_negative += 1-nn.functional.sigmoid(predictions).mean().item()
            total_negative += 1

        validation_loss /= total_positive + total_negative
        validation_accuracy_positive /= total_positive
        validation_accuracy_negative /= total_negative
        print(f"Validation: Loss = {validation_loss}, Accuracy positive = {validation_accuracy_positive}, Accuracy negative {validation_accuracy_negative}", flush=True)


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
                "train_accuracy_positive": train_accuracy_positive,
                "train_accuracy_negative": train_accuracy_negative,
                "validation_loss": validation_loss,
                "validation_accuracy_positive": validation_accuracy_positive,
                "validation_accuracy_negative": validation_accuracy_negative,
            }, f, indent=4)

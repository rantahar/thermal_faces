import os
import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from reduce_model import FaceDetector

data_path = "data/train_data"
batch_size = 1000
learning_rate = 1e-3
region_size = 32
region_step_fraction = 0.5
save_every = 500
num_epochs = 10001
units = 16

print("units:", units)

save_path = f"saved/reduction_model_6_{units}"

if not os.path.exists("saved"):
    os.makedirs("saved")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

def display_image(temperature_image):
    plt.imshow(temperature_image, cmap='hot')
    plt.axis('off')
    plt.title('Temperature Image')
    plt.show()


def extract_subregions(array, labels, height, width, step_fraction):
    subregions = []
    array_height, array_width = array.shape
    step_size = int(min(height, width) * step_fraction)

    for y in range(0, array_height - height + 1, step_size):
        for x in range(0, array_width - width + 1, step_size):
            subregion = np.array(array[y:y+height, x:x+width])
            contains_label = any(
                label["x"] >= x and label["x"] < x+width and
                label["y"] >= y and label["y"] < y+height
                for label in labels
            )
            subregions.append((subregion, contains_label))

    return subregions


def batch_data(data):
    arrays = []
    labels = []
    batches = []

    for i, (array, label) in enumerate(data):
        arrays.append(array)
        labels.append(label)

        if len(arrays) == batch_size or i == (len(data)-1):
            arrays_tensor = torch.tensor(np.array(arrays), dtype=torch.float32).to(device)
            labels_tensor = torch.tensor(labels, dtype=torch.float32).to(device)

            batches.append((arrays_tensor, labels_tensor))

            arrays = []
            labels = []
    
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

    train_data = []
    valid_data = []

    print("Creating subregions")
    for resolution in train_data_by_resolution:
        for frame_index, array, json_data in train_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]

            train_data += extract_subregions(
                array, labels, region_size, region_size, region_step_fraction
            )

    for resolution in valid_data_by_resolution:
        for frame_index, array, json_data in valid_data_by_resolution[resolution]:
            labels = [l for l in json_data if l['l'] == 1]

            valid_data += extract_subregions(
                array, labels, region_size, region_size, region_step_fraction
            )

    print("Creating batches")
    train_data = batch_data(train_data)
    valid_data = batch_data(valid_data)
            
    return train_data, valid_data

train_data, valid_data = load_npy_files(data_path)
print(f"Training data: {len(train_data)}, Validation data: {len(valid_data)}")

regions = 0
labels = 0
for batch in train_data:
    regions += batch[1].shape[0]
    labels += torch.sum(batch[1]).item()

print("Positive label fraction:", regions / labels)

# for batch in train_data:
#     print(batch[0].shape)
#     for i in range(batch_size):
#         print(batch[1][i].item())
#         display_image(batch[0][i].numpy())


loss_function = nn.BCEWithLogitsLoss()

def compute_loss(predictions, labels):
    predictions = predictions.view(-1)
    labels = labels.view(-1)
    loss = loss_function(predictions, labels)
    return loss


def calculate_accuracy(predictions, labels):
    predictions = nn.functional.sigmoid(predictions)
    accuracy = 1.-(labels-predictions).abs()
    average = accuracy.mean().item()
    return average


faceDetector = FaceDetector(region_size, region_size, units).to(device)
optimizer = torch.optim.Adam(faceDetector.parameters(), lr=learning_rate)



for epoch in range(num_epochs):
    faceDetector.train()
    batch_num = 0
    train_loss = 0
    accuracy = 0
    start_time = time.time()
    for data in train_data:
        frames = data[0]
        targets = data[1]

        predictions = faceDetector(frames)
        loss = compute_loss(predictions, targets)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        accuracy += calculate_accuracy(predictions, targets)
        batch_num += 1

    batch_time = time.time() - start_time
    train_loss /= batch_num
    accuracy /= batch_num
    print(f"Epoch {epoch}: Loss = {train_loss}, Accuracy = {accuracy}, Time: {batch_time:.2f}s")

    validation_loss = 0
    validation_accuracy = 0
    total = 0
    with torch.no_grad():
        for data in valid_data:
            frames = data[0]
            targets = data[1]

            predictions = faceDetector(frames)
            loss = compute_loss(predictions, targets)
            validation_loss += loss.item()

            validation_accuracy += calculate_accuracy(predictions, targets)
            total += 1

        validation_loss /= total
        validation_accuracy /= total
        print(f"Validation: Loss = {validation_loss}, Accuracy = {validation_accuracy}", flush=True)


    if epoch%save_every == 0:
        model_state = faceDetector.state_dict()
        torch.save(
            {
                'model_state_dict': faceDetector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            },
            f"{save_path}_{epoch}"
        )
        with open(f"{save_path}_{epoch}.json", 'w') as f:
            json.dump({
                'epoch': epoch,
                "train_loss": train_loss,
                "train_accuracy": accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }, f, indent=4)

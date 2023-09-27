import os
import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

from simple_model import FaceDetector

data_path = "data/train_data"
batch_size = 5
learning_rate = 1e-3
label_loss_weight = 1e3
label_weight_decay = 0.996
min_label_weight = 10
save_every = 500
num_epochs = 10001
label_fuzzyness = 0
units = 8

save_path = f"saved/model_5_{units}"

if not os.path.exists("saved"):
    os.makedirs("saved")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

def display_image(temperature_image, target_image):
    plt.imshow(temperature_image, cmap='hot')
    plt.axis('off')
    plt.title('Temperature Image')

    target_coords = np.argwhere(target_image != 0)
    for coord in target_coords:
        row, col = coord
        plt.plot(col, row, 'b+', markersize=1)

    plt.show()

def display_image_target(temperature_image, target_image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(temperature_image, cmap='hot')
    axs[0].axis('off')
    axs[0].set_title('Temperature Image')

    axs[1].imshow(target_image, cmap='binary')
    axs[1].axis('off')
    axs[1].set_title('Target Image')

    plt.show()


def mark_neighboring_pixels(original_temperature, temperature_image, target_image, x, y, threshold, max_dist):
    if max_dist < 1:
        return
    
    height, width = temperature_image.shape
    
    # Set the target pixel at the starting position to 1
    target_image[y, x] = 1
    
    # Check neighboring pixels
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue
            
            # Calculate neighboring pixel coordinates
            nx = x + i
            ny = y + j

            # Skip if already set
            if target_image[ny, nx] == 1:
                continue

            # Check if neighboring pixel is within image bounds
            if ny >= 0 and ny < height and nx >= 0 and nx < width:
                # Check if temperature difference is below threshold
                if abs(temperature_image[ny, nx] - original_temperature) < threshold:
                    # Recursively mark neighboring pixels
                    mark_neighboring_pixels(original_temperature, temperature_image, target_image, nx, ny, threshold, max_dist-1)


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
            
            labeled_array = np.zeros_like(array)
            for label in json_data['labels']:
                x = label['x']
                y = label['y']
                l = label['l']
                if l == 1:
                    labeled_array[y, x] = 1
                    #mark_neighboring_pixels(array[y, x], array, labeled_array, x, y, 0.1, 4)
            labeled_array = labeled_array*(1-2*label_fuzzyness) + label_fuzzyness
            
            # display_image_target(array, labeled_array)
            file_name = array.shape
            if file_name in data:
                data[file_name].append((frame_index, array, labeled_array))
                data[file_name].append((frame_index+0.5, np.flip(array, axis=0), np.flip(labeled_array, axis=0)))
            else:
                data[file_name] = [
                    (frame_index, array, labeled_array),
                    (frame_index+0.5, np.flip(array, axis=0), np.flip(labeled_array, axis=0)),
                ]
    
    for key in data.keys():
        data[key].sort(key=lambda x: x[0])
    
    train_data_by_resolution = {}
    valid_data_by_resolution = {}

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

    for resolution in train_data_by_resolution:
        data = train_data_by_resolution[resolution]
        frames = np.stack([frame[1] for frame in data], axis=0)
        labels = np.stack([frame[2] for frame in data], axis=0)
        num_batches = np.max([1, frames.shape[0] // batch_size])
        for i in range(num_batches+1):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if start_idx < frames.shape[0]:
                data = torch.from_numpy(frames[start_idx:end_idx]).float().to(device)
                target = torch.from_numpy(labels[start_idx:end_idx]).float().to(device)
                train_data.append((data,target))
        
    for resolution in valid_data_by_resolution:
        data = valid_data_by_resolution[resolution]
        frames = np.stack([frame[1] for frame in data], axis=0)
        labels = np.stack([frame[2] for frame in data], axis=0)
        num_batches = np.max([1, frames.shape[0] // batch_size])
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size
            if start_idx < frames.shape[0]:
                data = torch.from_numpy(frames[start_idx:end_idx]).float().to(device)
                target = torch.from_numpy(labels[start_idx:end_idx]).float().to(device)
                valid_data.append((data,target))
            
    return train_data, valid_data

train_data, valid_data = load_npy_files(data_path)
print(len(train_data), len(valid_data))
targets = 0
pixels = 0
for batch in train_data:
    pixels += batch[1].shape[0]*batch[1].shape[1]*batch[1].shape[2]
    targets += torch.sum(batch[1]).item()
print(pixels, targets, pixels / targets)


loss_function = nn.BCEWithLogitsLoss(reduction='none')

def compute_loss(predictions, labels):
    #predictions = nn.functional.sigmoid(predictions)
    batch_size = labels.size(0)

    predictions = predictions.view(-1)
    labels = labels.view(-1)
    weights = torch.ones_like(labels)
    weights[labels > 0.5] = label_loss_weight
    loss = loss_function(predictions, labels)
    loss = torch.sum(loss[labels > 0.5]) * label_loss_weight + torch.sum(loss[labels < 0.5]) 
    return loss


def calculate_accuracy(predictions, labels):
    predictions = nn.functional.sigmoid(predictions)
    average_at_label = predictions[labels > 0.5].mean().item()
    average = 1.-(labels-predictions).abs().mean().item()
    return average, average_at_label


faceDetector = FaceDetector(units).to(device)
optimizer = torch.optim.Adam(faceDetector.parameters(), lr=learning_rate)



for epoch in range(num_epochs):

    faceDetector.train()
    batch_num = 0
    train_loss = 0
    accuracy = 0
    accuracy_label = 0
    start_time = time.time()
    for data in train_data:
        frames = data[0]
        targets = data[1]

        predictions = faceDetector(frames)
        loss = compute_loss(predictions, targets)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        acc, acc_label = calculate_accuracy(predictions, targets)
        accuracy_label += acc_label
        accuracy += acc
        batch_num += 1

    batch_time = time.time() - start_time
    train_loss /= batch_num
    accuracy_label /= batch_num
    accuracy /= batch_num
    print(f"Epoch {epoch}: Loss = {train_loss}, Accuracy = {accuracy}, Accuracy at label = {accuracy_label}, label weight {label_loss_weight:.0f}, Time: {batch_time:.2f}s")

    validation_loss = 0
    validation_accuracy = 0
    validation_accuracy_label = 0
    total = 0
    with torch.no_grad():
        for data in valid_data:
            frames = data[0]
            targets = data[1]

            predictions = faceDetector(frames)
            loss = compute_loss(predictions, targets)
            validation_loss += loss.item()

            accuracy, accuracy_label = calculate_accuracy(predictions, targets)
            validation_accuracy_label += accuracy_label
            validation_accuracy += accuracy
            total += 1

        validation_loss /= total
        validation_accuracy_label /= total
        validation_accuracy /= total
        print(f"Validation: Loss = {validation_loss}, Accuracy = {validation_accuracy}, Accuracy at label = {validation_accuracy_label}", flush=True)

    if label_loss_weight > min_label_weight:
        label_loss_weight *= label_weight_decay
    else:
        label_loss_weight = min_label_weight

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
                "train_accuracy_label": accuracy_label,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
                "validation_accuracy_label": validation_accuracy_label
            }, f, indent=4)

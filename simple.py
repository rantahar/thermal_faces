import os
import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


data_path = "data/train_data"
batch_size = 2
learning_rate = 0.001
label_loss_weight = 100000
num_epochs = 5

save_path = f"saved/model"

device = "cuda" if torch.cuda.is_available() else "cpu"


def display_image(temperature_image, target_image):
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(temperature_image, cmap='hot')
    axs[0].axis('off')
    axs[0].set_title('Temperature Image')

    axs[1].imshow(target_image, cmap='binary')
    axs[1].axis('off')
    axs[1].set_title('Target Image')

    plt.show()


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

            if video_name in data:
                data[video_name].append((frame_index, array, labeled_array))
            else:
                data[video_name] = [(frame_index, array, labeled_array)]
    
    for key in data.keys():
        data[key].sort(key=lambda x: x[0])
    
    train_data = {}
    val_data = {}

    for video_name, frames in data.items():
        if len(frames) > 2:
            num_frames = len(frames)
            start_frame = int(np.floor((0.5-0.5*validation_fraction)*num_frames))
            end_frame = int(np.ceil((0.5+0.5*validation_fraction)*num_frames))

            val_frames = frames[start_frame:end_frame]
            val_data[video_name] = (
                np.stack([frame[1] for frame in val_frames], axis=0),
                np.stack([frame[2] for frame in val_frames], axis=0)
            )

            train_frames = frames[:start_frame] + frames[end_frame:]
            train_data[video_name] = (
                np.stack([frame[1] for frame in train_frames], axis=0),
                np.stack([frame[2] for frame in train_frames], axis=0)
            )

    return train_data, val_data


train_data, val_data = load_npy_files(data_path)

print()
for video in train_data:
    print(video, train_data[video][0].shape, train_data[video][1].shape)
print()
for video in val_data:
    print(video, val_data[video][0].shape, val_data[video][1].shape)


class FaceDetector(nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, temperatures):
        temperatures = temperatures.unsqueeze(1)

        out = self.conv1(temperatures)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.sigmoid(out)

        out = out.squeeze(1)
        return out


def compute_loss(predictions, targets):
    #predictions = torch.sigmoid(predictions)
    batch_size = predictions.size(0)

    predictions = predictions.view(-1)
    targets = targets.view(-1)
    weights = torch.ones_like(targets)
    weights[targets == 1] = label_loss_weight
    loss = nn.functional.binary_cross_entropy(predictions, targets, weight=weights, reduction='sum')
    return loss / batch_size

def calculate_accuracy(predictions, labels):
    average_at_label = predictions[labels == 1].mean().item()
    average_other = predictions[labels == 0].mean().item()
    return average_at_label, 1-average_other


faceDetector = FaceDetector().to(device)
optimizer = torch.optim.Adam(faceDetector.parameters(), lr=learning_rate)


for epoch in range(num_epochs):

    faceDetector.train()
    batch_num = 0
    for video in train_data:
        frames = train_data[video][0]
        labels = train_data[video][1]
        num_batches = frames.shape[0] // batch_size
        data_batches = np.array_split(frames, num_batches)
        label_batches = np.array_split(labels, num_batches)
        for batch, labels in zip(data_batches, label_batches):
            batch_num += 1

            batch = torch.from_numpy(batch).float().to(device)
            targets = torch.from_numpy(labels).float().to(device)

            predictions = faceDetector(batch)
            loss = compute_loss(predictions, targets)

            loss.backward()
            optimizer.step()

            accuracy_label, accuracy_other = calculate_accuracy(predictions, targets)

            print(f"Batch {batch_num}: Loss = {loss.item()}, Accuracy at label = {accuracy_label}, Accuracy otherwise = {accuracy_other}")

    faceDetector.eval()
    validation_loss = 0
    validation_accuracy_label = 0
    validation_accuracy_other = 0
    total = 0
    with torch.no_grad():
        for video in val_data:
            frames = val_data[video][0]
            labels = val_data[video][1]

            frames = torch.from_numpy(frames).float().to(device)
            targets = torch.from_numpy(labels).float().to(device)

            predictions = faceDetector(frames)
            loss = compute_loss(predictions, targets)
            validation_loss += loss.item()

            accuracy_label, accuracy_other = calculate_accuracy(predictions, targets)
            validation_accuracy_label += accuracy_label
            validation_accuracy_other += accuracy_other
            total += 1

        validation_loss /= total
        validation_accuracy_label /= total
        validation_accuracy_other /= total
        print(f"Validation: Loss = {validation_loss}, Accuracy at label = {validation_accuracy_label}, Accuracy otherwise = {validation_accuracy_other}")

    model_state = faceDetector.state_dict()
    save_dict = {
        "model_state": model_state,
        "validation_loss": validation_loss,
        "validation_accuracy_label": validation_accuracy_label,
        "validation_accuracy_other": validation_accuracy_other
    }
    with open(f"{save_path}_{epoch}", "w") as f:
        json.dump(save_dict, f)



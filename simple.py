import os
import numpy as np
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import time

data_path = "data/train_data"
batch_size = 5
learning_rate = 1e-4
label_loss_weight = 2e3
save_every = 500
num_epochs = 5000
units = 16

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


def mark_neighboring_pixels(original_temperature, temperature_image, target_image, x, y, threshold):
    height, width = temperature_image.shape
    
    # Set the target pixel at the starting position to 1
    target_image[y, x] = 1
    
    # Check neighboring pixels
    for i in range(-1, 2):
        for j in range(-1, 2):
            # Skip if current position is the starting position
            if i == 0 and j == 0:
                continue
            
            # Calculate neighboring pixel coordinates
            nx = x + i
            ny = y + j
            
            # Check if neighboring pixel is within image bounds
            if ny >= 0 and ny < height and nx >= 0 and nx < width:
                # Check if temperature difference is below threshold
                if abs(temperature_image[ny, nx] - original_temperature) < threshold and target_image[ny, nx] != 1:
                    # Recursively mark neighboring pixels
                    mark_neighboring_pixels(original_temperature, temperature_image, target_image, nx, ny, threshold)


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
                    mark_neighboring_pixels(array[y, x], array, labeled_array, x, y, 0.2)

            display_image_target(array, labeled_array)

            if video_name in data:
                data[video_name].append((frame_index, array, labeled_array))
            else:
                data[video_name] = [(frame_index, array, labeled_array)]
    
    for key in data.keys():
        data[key].sort(key=lambda x: x[0])
    
    train_data = []
    valid_data = []

    for video_name, frames in data.items():
        if len(frames) > 2:
            num_frames = len(frames)
            start_frame = int(np.floor((0.5-0.5*validation_fraction)*num_frames))
            end_frame = int(np.ceil((0.5+0.5*validation_fraction)*num_frames))

            val_frames = frames[start_frame:end_frame]
            train_frames = frames[:start_frame] + frames[end_frame:]

            frames = np.stack([frame[1] for frame in val_frames], axis=0)
            labels = np.stack([frame[2] for frame in val_frames], axis=0)
            num_batches = np.max([1, frames.shape[0] // batch_size])
            for i in range(num_batches+1):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                if start_idx < frames.shape[0]:
                    data = torch.from_numpy(frames[start_idx:end_idx]).float().to(device)
                    target = torch.from_numpy(labels[start_idx:end_idx]).float().to(device)
                    valid_data.append((data,target))

            frames = np.stack([frame[1] for frame in train_frames], axis=0)
            labels = np.stack([frame[2] for frame in train_frames], axis=0)
            num_batches = np.max([1, frames.shape[0] // batch_size])
            for i in range(num_batches+1):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                if start_idx < frames.shape[0]:
                    data = torch.from_numpy(frames[start_idx:end_idx]).float().to(device)
                    target = torch.from_numpy(labels[start_idx:end_idx]).float().to(device)
                    train_data.append((data,target))
            
    return train_data, valid_data


train_data, valid_data = load_npy_files(data_path)


class FaceDetector(nn.Module):
    def __init__(self):
        super(FaceDetector, self).__init__()
        self.conv1 = nn.Conv2d(1, units, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(units, 2*units, kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(2*units, 2*units, kernel_size=3, stride=1, padding=1)
        #self.conv4 = nn.Conv2d(4*units, 2*units, kernel_size=3, stride=1, padding=1)
        #self.conv5 = nn.Conv2d(4*units, 2*units, kernel_size=3, stride=1, padding=1)

        self.conv6 = nn.ConvTranspose2d(2*units, units, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv7 = nn.ConvTranspose2d(units, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
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
        #out = self.conv4(out)
        #out = self.relu(out)
        #out = self.conv5(out)
        #out = self.relu(out)
        out = self.conv6(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.sigmoid(out)
        
        out = out.squeeze(1)
        return out


def compute_loss(predictions, targets):
    #predictions = torch.sigmoid(predictions)
    batch_size = predictions.size(0)

    predictions = predictions.view(-1)
    targets = targets.view(-1)
    weights = torch.ones_like(targets)
    weights[targets == 0] /= label_loss_weight
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
    train_loss = 0
    accuracy_label = 0
    accuracy_other = 0
    start_time = time.time()
    for data in train_data:
        frames = data[0]
        targets = data[1]

        #image_array = frames[-1].numpy()
        #target_array = targets[-1].numpy()
        #display_image(image_array, target_array)

        predictions = faceDetector(frames)
        #print(predictions[0].cpu().detach().numpy())
        loss = compute_loss(predictions, targets)

        loss.backward()
        optimizer.step()

        acc_label, acc_other = calculate_accuracy(predictions, targets)
        train_loss += loss.item()
        accuracy_label += acc_label
        accuracy_other += acc_other
        batch_num += 1

    batch_time = time.time() - start_time
    print(f"Epoch {epoch}: Loss = {loss/batch_num}, Accuracy at label = {accuracy_label/batch_num}, Accuracy otherwise = {accuracy_other/batch_num}, Time: {batch_time:.2f}s")

    faceDetector.eval()
    validation_loss = 0
    validation_accuracy_label = 0
    validation_accuracy_other = 0
    total = 0
    with torch.no_grad():
        for data in valid_data:
            frames = data[0]
            targets = data[1]

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
        print(f"Validation: Loss = {validation_loss}, Accuracy at label = {validation_accuracy_label}, Accuracy otherwise = {validation_accuracy_other}", flush=True)

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
                "validation_loss": validation_loss,
                "validation_accuracy_label": validation_accuracy_label,
                "validation_accuracy_other": validation_accuracy_other
            }, f, indent=4)

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

batch_size = 100
learning_rate = 1e-3
negatives_per_positive = 1
region_size = 48
save_every = 100
num_epochs = 2001
units = 16

print("units:", units)

save_path = f"saved/reduction_model_{units}_{region_size}"

if not os.path.exists("saved"):
    os.makedirs("saved")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


valid_data_positive = torch.load("valid_data_positive.pt")
valid_data_negative = torch.load("valid_data_negative.pt")
train_data_positive = torch.load("train_data_positive.pt")
train_data_negative = torch.load("train_data_negative.pt")
valid_data_positive = valid_data_positive.to(device)
valid_data_negative = valid_data_negative.to(device)
train_data_positive = train_data_positive.to(device)
train_data_negative = train_data_negative.to(device)
print(f"Positive training data: {len(train_data_positive)}, negative: {len(train_data_negative)}")
print(f"Positive validation data: {len(valid_data_positive)}, negative: {len(valid_data_negative)}")



faceDetector = FaceDetector(region_size, region_size, units).to(device)
optimizer = torch.optim.Adam(faceDetector.parameters(), lr=learning_rate)
loss_function = nn.BCEWithLogitsLoss()

# Cyclical iterator to get batches of negative examples. Helps in balancing
# positive and negative examples when there are many more negative ones.
negative_train_data = itertools.cycle(train_data_negative)


for epoch in range(num_epochs):
    faceDetector.train()
    negative_batches = 0
    positive_batches = 0
    train_loss = 0
    train_accuracy_positive = 0
    train_accuracy_negative = 0
    start_time = time.time()
    for i in range(train_data_positive.shape[0]):
        data = train_data_positive[i,:,:,:]
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
        for i in range(valid_data_positive.shape[0]):
            data = valid_data_positive[i,:,:,:]
            predictions = faceDetector(data)
            validation_loss += loss_function(predictions, torch.ones_like(predictions)).item()
            validation_accuracy_positive += nn.functional.sigmoid(predictions).mean().item()
            total_positive += 1
        
        for data in valid_data_negative:
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
        


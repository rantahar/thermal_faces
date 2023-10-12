import os
import click
import json
import torch
import torch.nn as nn
import time
import itertools

from thermal_face_detector.reduce_model import FaceDetector

device = "cuda" if torch.cuda.is_available() else "cpu"

# Region size of the regions after rescaling. By default, regions are rescaled
# to the smallest region size
@click.command()
@click.option("--units", default=32, help="Hidden units after the first layer. Used to rescale the model.")
@click.option("--region_size", default=32, help="The size of the training images (generate using process_data.py)")
@click.option("--num_epochs", default=1000, help="Number of epochs to run")
@click.option("--save_every", default=50, help="Number of epochs between saving the model")
@click.option("--learning_rate", default=1e-5, help="Optimization step size")
@click.option("--negatives", default=1, help="Number of negative examples for each positive")
def train_subsection_model(units, region_size, num_epochs, save_every, learning_rate, negatives):

    save_path = f"saved/reduction_model_{units}_{region_size}_{negatives}"

    if not os.path.exists("saved"):
        os.makedirs("saved")

    print("units:", units)
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
        batches = 0
        train_loss = 0
        train_accuracy = 0
        std = 0
        start_time = time.time()
        for i in range(train_data_positive.shape[0]):
            data = train_data_positive[i,:,:,:]
            positive_predictions = faceDetector(data)
            loss = loss_function(positive_predictions, torch.ones_like(positive_predictions))
            train_accuracy += nn.functional.sigmoid(positive_predictions).mean().item()
            predictions = nn.functional.sigmoid(positive_predictions).detach()

            for i in range(negatives):
                negative_data = next(negative_train_data)
                negative_predictions = faceDetector(negative_data)
                loss += loss_function(negative_predictions, torch.zeros_like(negative_predictions))
                train_accuracy += 1 - nn.functional.sigmoid(negative_predictions).mean().item()
                predictions = torch.cat([predictions, nn.functional.sigmoid(negative_predictions).detach()])
                batches += 1

            std += predictions.cpu().numpy().std().item()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

        batch_time = time.time() - start_time
        train_loss /= batches
        train_accuracy /= batches
        std /= train_data_positive.shape[0]
        print(f"Epoch {epoch}: Loss = {train_loss}, Accuracy = {train_accuracy}, std = {std}, Time: {batch_time:.2f}s")

        if std < 1e-5:
          print("Early stopping due to low result standard deviation.")
          break

        validation_loss = 0
        validation_accuracy = 0
        total = 0
        with torch.no_grad():
            for i in range(valid_data_positive.shape[0]):
                data = valid_data_positive[i,:,:,:]
                predictions = faceDetector(data)
                validation_loss += loss_function(predictions, torch.ones_like(predictions)).item()
                validation_accuracy += nn.functional.sigmoid(positive_predictions).mean().item()
                total += 1

            for data in valid_data_negative:
                predictions = faceDetector(data)
                validation_loss += loss_function(predictions, torch.zeros_like(predictions)).item()
                validation_accuracy += 1 - nn.functional.sigmoid(negative_predictions).mean().item()
                total += 1

            validation_loss /= total
            validation_accuracy /= total
            print(f"Validation: Loss = {validation_loss}, Accuracy = {validation_accuracy}", flush=True)


        if (epoch+1)%save_every == 0:
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
        


if __name__ == '__main__':
    train_subsection_model()

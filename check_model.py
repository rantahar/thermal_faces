import sys
import torch
from reduce_model import FaceDetector

path_to_model = sys.argv[1]

model_dict = torch.load(
    path_to_model,
    map_location=torch.device('cpu')
)
units = model_dict.get("units")
region_size = model_dict.get("region_size")

model = FaceDetector(region_size, region_size, units)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

valid_data_positive = torch.load("valid_data_positive.pt")
valid_data_negative = torch.load("valid_data_negative.pt")
train_data_positive = torch.load("train_data_positive.pt")
train_data_negative = torch.load("train_data_negative.pt")



accuracy = 0
for i in range(valid_data_negative.shape[0]):
    result = model(valid_data_negative[i,:,:,:])
    accuracy += result.mean().item()

print("valid_data_positive", accuracy / valid_data_negative.shape[0])

accuracy = 0
for i in range(valid_data_positive.shape[0]):
    result = model(valid_data_positive[i,:,:,:])
    accuracy += result.mean().item()

print("valid_data_negative", accuracy / valid_data_positive.shape[0])


accuracy = 0
for i in range(train_data_negative.shape[0]):
    result = model(train_data_negative[i,:,:,:])
    accuracy += result.mean().item()

print("train_data_positive", accuracy / train_data_negative.shape[0])

accuracy = 0
for i in range(train_data_positive.shape[0]):
    result = model(train_data_positive[i,:,:,:])
    accuracy += result.mean().item()

print("train_data_negative", accuracy / train_data_positive.shape[0])



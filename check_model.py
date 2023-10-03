import sys
import torch
from reduce_model import FaceDetector

path_to_model = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


model_dict = torch.load(
    path_to_model,
    map_location=torch.device(device)
)
units = model_dict.get("units")
region_size = model_dict.get("region_size")

model = FaceDetector(region_size, region_size, units)
model.load_state_dict(model_dict['model_state_dict'])
model.to(device)
model.eval()

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



result = model(valid_data_negative.reshape((-1,region_size,region_size)))
accuracy = 1-result.mean().item()
print("valid_data_negative", accuracy)

result = model(valid_data_positive.reshape((-1,region_size,region_size)))
accuracy = result.mean().item()
print("valid_data_positive", accuracy)


result = model(train_data_negative.reshape((-1,region_size,region_size)))
accuracy = 1-result.mean().item()
print("train_data_negative", accuracy)

result = model(train_data_positive.reshape((-1,region_size,region_size)))
accuracy = result.mean().item()
print("train_data_positive", accuracy)



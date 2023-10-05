import sys
import torch
import json
import numpy as np
from subsection_utils import extract_subregions, plot_boxes_on_image, non_max_suppression
from reduce_model import FaceDetector


threshold = 0.9
units = 16
region_size = 64
step_fraction = 0.2
input_image = "data/train_data/Pilot2_MagicFlute_Part1_18500.npy"
if len(sys.argv) > 1:
    path_to_model = sys.argv[1]
else:
    path_to_model = f"saved/reduction_model_6_{units}_1000"

# Load the model from disk
model_dict = torch.load(
    path_to_model,
    map_location=torch.device('cpu')
)
units = model_dict.get("units", units)
region_size = model_dict.get("region_size", region_size)

model = FaceDetector(region_size, region_size, units)
model.load_state_dict(model_dict['model_state_dict'])
model.eval()

json_file = input_image.replace(".npy",".json")

# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
with open(json_file, 'r') as f:
    json_data = json.load(f)


# Load and preprocess the temperature values
image = np.load(input_image).astype(np.float32)
regions = extract_subregions(image, json_data["labels"], region_size, region_size, step_fraction)
arrays = [r[0] for r in regions]
labels = [r[1] for r in regions]

with torch.no_grad():
    tensor = torch.tensor(arrays)
    predictions = model(tensor).squeeze(0).numpy()

print(predictions.max())

# First apply treshold to remove most of the boxes
boxes = []
threshold = threshold * predictions.max()
for i in range(len(regions)):
    array, _, x, y = regions[i]
    output = predictions[i]

    if output > threshold:
        boxes.append((x, y, region_size, region_size, output))
    
boxes = non_max_suppression(boxes, 0.05)

plot_boxes_on_image(image, boxes)



